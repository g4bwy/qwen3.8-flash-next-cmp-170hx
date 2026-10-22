import ctypes, os, random, struct, time
import numpy as np

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                      ctypes.c_int, ctypes.c_int, ctypes.c_long]
L = ctypes.c_long
SYS_setup, SYS_enter = 425, 426
IORING_OP_READ = 22
CQE_SZ = 16
OFF_SQ_RING, OFF_CQ_RING, OFF_SQES = 0, 0x8000000, 0x10000000


class _SO(ctypes.Structure):
    _fields_ = [("head", ctypes.c_uint32), ("tail", ctypes.c_uint32),
                ("ring_mask", ctypes.c_uint32), ("ring_entries", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("dropped", ctypes.c_uint32),
                ("array", ctypes.c_uint32), ("resv1", ctypes.c_uint32),
                ("user_addr", ctypes.c_uint64)]


class _CO(ctypes.Structure):
    _fields_ = [("head", ctypes.c_uint32), ("tail", ctypes.c_uint32),
                ("ring_mask", ctypes.c_uint32), ("ring_entries", ctypes.c_uint32),
                ("overflow", ctypes.c_uint32), ("cqes", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("resv1", ctypes.c_uint32),
                ("user_addr", ctypes.c_uint64)]


class Params(ctypes.Structure):
    _fields_ = [("sq_entries", ctypes.c_uint32), ("cq_entries", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("sq_thread_cpu", ctypes.c_uint32),
                ("sq_thread_idle", ctypes.c_uint32), ("features", ctypes.c_uint32),
                ("wq_fd", ctypes.c_uint32), ("resv", ctypes.c_uint32 * 3),
                ("sq_off", _SO), ("cq_off", _CO)]


# numpy structured dtype == struct io_uring_sqe (64 bytes)
SQE_DTYPE = np.dtype([
    ("opcode", np.uint8), ("flags", np.uint8), ("ioprio", np.uint16),
    ("fd", np.int32), ("off", np.uint64), ("addr", np.uint64),
    ("len", np.uint32), ("rw_flags", np.uint32), ("user_data", np.uint64),
    ("pad", np.uint8, 24),   # buf_index/personality/addr3/__pad2 to fill 64
])
assert SQE_DTYPE.itemsize == 64, SQE_DTYPE.itemsize


class Ring:
    """Vectorized batched scattered READ: SQE table filled with numpy, not ctypes."""

    def __init__(self, entries=512):
        self.p = Params()
        self.fd = libc.syscall(L(SYS_setup), L(entries), ctypes.byref(self.p))
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "io_uring_setup")
        n = self.p.sq_entries
        self.n = n
        self.sq_ring = self._map(OFF_SQ_RING, self.p.sq_off.array + n * 4)
        self.cq_ring = self._map(OFF_CQ_RING,
                                 self.p.cq_off.cqes + self.p.cq_entries * CQE_SZ)
        self.sqes = self._map(OFF_SQES, n * 64)
        o = self.p.sq_off
        self.sq_tail = self._u32(self.sq_ring, o.tail)
        self.mask = self._u32(self.sq_ring, o.ring_mask).value
        self.sq_array = (ctypes.addressof(self.sq_ring) + o.array)
        c = self.p.cq_off
        self.cq_mask = self._u32(self.cq_ring, c.ring_mask).value
        self.cq_head = self._u32(self.cq_ring, c.head)
        self.cq_tail = self._u32(self.cq_ring, c.tail)
        self.cqes = ctypes.addressof(self.cq_ring) + c.cqes
        self.cqes_off = c.cqes
        self.sqe_view = np.frombuffer(self.sqes, dtype=SQE_DTYPE)  # 0..n-1 slots

    def close(self):
        os.close(self.fd)

    @staticmethod
    def _u32(buf, off):
        return ctypes.c_uint32.from_buffer(buf, off)

    def _map(self, pgoff, length):
        a = libc.mmap(None, length, 3, 1, self.fd, pgoff)
        if a is None or a == ctypes.c_void_p(-1).value:
            raise OSError("mmap ring failed")
        return (ctypes.c_ubyte * length).from_address(a)

    def gather(self, fd, addrs, offs, lengths):
        """addrs/offs/lengths: equal-length numpy int64/int32 arrays."""
        m = len(addrs)
        start = self.sq_tail.value
        idx = (start + np.arange(m)) & self.mask
        v = self.sqe_view[idx]           # gather existing slots (view order)
        v["opcode"] = IORING_OP_READ
        v["fd"] = fd
        v["off"] = offs
        v["addr"] = addrs
        v["len"] = lengths
        v["user_data"] = np.arange(m, dtype=np.uint64)
        self.sqe_view[idx] = v           # scatter back to the ring
        # sq_array[j] = ring slot index (tail&mask); io_uring indirection
        sqarr = np.frombuffer(
            (ctypes.c_ubyte * (4 * self.n)).from_address(self.sq_array),
            dtype=np.uint32)
        sqarr[idx] = idx
        self.sq_tail.value = start + m
        if libc.syscall(L(SYS_enter), L(self.fd), L(m), L(m), L(1), L(0), L(0)) < 0:
            raise OSError(ctypes.get_errno(), "io_uring_enter")
        results = np.frombuffer(
            (ctypes.c_ubyte * (self.cq_mask + 1) * 16).from_address(self.cqes),
            dtype=np.int32)
        outs = [None] * m
        while self.cq_head.value != self.cq_tail.value:
            e = (self.cq_head.value & self.cq_mask) * 16
            ui = int.from_bytes(
                bytes(self.cq_ring[self.cqes_off + e:self.cqes_off + e + 8]),
                "little")
            outs[ui] = results[(e + 8) // 4]
            self.cq_head.value += 1
        return outs


if __name__ == "__main__":
    ROW = 2048
    random.seed(7)
    FILE_ROWS = 100_000
    with open("/tmp/uring_table.bin", "wb") as f:
        f.write(b"".join(struct.pack("<I", r) + b"\0" * (ROW - 4)
                         for r in range(FILE_ROWS)))
    fd = os.open("/tmp/uring_table.bin", os.O_RDONLY)
    for batch in (1024, 8192, 32768):
        offs_r = np.array(random.sample(range(FILE_ROWS), batch), dtype=np.int64)
        bufs = [ctypes.create_string_buffer(ROW) for _ in range(batch)]
        addrs = np.array([ctypes.addressof(b) for b in bufs], dtype=np.uint64)
        flen = np.full(batch, ROW, dtype=np.uint32)
        foff = (offs_r * ROW).astype(np.uint64)
        for o in foff:
            os.pread(fd, ROW, int(o))                     # warm cache
        t0 = time.perf_counter()
        for b, o in zip(bufs, foff):
            os.preadv(fd, [b], int(o))                     # preadv C loop
        t_preadv = (time.perf_counter() - t0) * 1e3
        for b in bufs:
            b.raw = b"\0" * ROW
        ring = Ring(entries=max(1024, batch))
        t0 = time.perf_counter()
        res = ring.gather(fd, addrs, foff, flen)
        t_uring = (time.perf_counter() - t0) * 1e3
        ring.close()
        ok = all(r == ROW for r in res) and all(
            struct.unpack("<I", bufs[i].raw[:4])[0] == offs_r[i]
            for i in range(batch))
        print(f"batch={batch:6d} preadv={t_preadv:7.2f}ms uring={t_uring:7.2f}ms "
              f"speedup={t_preadv/max(t_uring,0.01):4.1f}x correct={ok}")
