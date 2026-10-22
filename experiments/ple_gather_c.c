/* PLE gather benchmarks: C preadv loop vs C io_uring, no Python in the loop.
 * Build: cc -O2 -shared -fPIC -o /tmp/ple_gather.so /tmp/ple_gather.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/io_uring.h>

/* fds/offs are caller-provided contiguous arrays; one row per entry.
 * Returns number of short reads (0 = all good). */
long gather_preadv(const int *fds, const long long *offs, char *base,
                   long row_bytes, long n) {
    long bad = 0;
    for (long i = 0; i < n; i++) {
        struct iovec iov = { .iov_base = base + i * row_bytes,
                             .iov_len = (size_t)row_bytes };
        ssize_t r = preadv(fds[i], &iov, 1, (off_t)offs[i]);
        bad += (r != row_bytes);
    }
    return bad;
}


/* io_uring gather: fill n READ SQEs, single enter, C loop reap.
 * async=1 sets IOSQE_ASYNC. fixed=1 registers fd table (fds become
 * indices after first dedup... simple mode: fixed files registered
 * upfront per call for the distinct small set). Returns short count or
 * -errno on setup failure. */
struct ring {
    int ring_fd;
    struct io_uring_params p;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    void *sq_map, *cq_map, *sqe_map;
    size_t sq_len, cq_len, sqe_len;
};

static int ring_init(struct ring *r, unsigned entries) {
    memset(&r->p, 0, sizeof(r->p));
    r->ring_fd = (int)syscall(__NR_io_uring_setup, entries, &r->p);
    if (r->ring_fd < 0) return -errno;
    r->sq_len = r->p.sq_off.array + r->p.sq_entries * sizeof(unsigned);
    r->cq_len = r->p.cq_off.cqes + r->p.cq_entries * sizeof(struct io_uring_cqe);
    r->sqe_len = (size_t)r->p.sq_entries * sizeof(struct io_uring_sqe);
    r->sq_map = mmap(0, r->sq_len, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, r->ring_fd, IORING_OFF_SQ_RING);
    r->cq_map = mmap(0, r->cq_len, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, r->ring_fd, IORING_OFF_CQ_RING);
    r->sqe_map = mmap(0, r->sqe_len, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, r->ring_fd, IORING_OFF_SQES);
    if (r->sq_map == MAP_FAILED || r->cq_map == MAP_FAILED ||
        r->sqe_map == MAP_FAILED) return -ENOMEM;
    unsigned *sqb = r->sq_map, *cqb = r->cq_map;
    r->sq_head = sqb + r->p.sq_off.head / 4;
    r->sq_tail = sqb + r->p.sq_off.tail / 4;
    r->sq_mask = sqb + r->p.sq_off.ring_mask / 4;
    r->sq_array = sqb + r->p.sq_off.array / 4;
    r->cq_head = cqb + r->p.cq_off.head / 4;
    r->cq_tail = cqb + r->p.cq_off.tail / 4;
    r->cq_mask = cqb + r->p.cq_off.ring_mask / 4;
    r->sqes = r->sqe_map;
    r->cqes = (struct io_uring_cqe *)((char *)cqb + r->p.cq_off.cqes);
    return 0;
}

static void ring_free(struct ring *r) {
    munmap(r->sq_map, r->sq_len);
    munmap(r->cq_map, r->cq_len);
    munmap(r->sqe_map, r->sqe_len);
    close(r->ring_fd);
}

long gather_uring(const int *fds, const long long *offs, char *base,
                  long row_bytes, long n, int async) {
    static struct ring r;
    static int inited = -1;
    static long cap;
    if (inited != 0 || n > cap) {
        if (inited == 0) ring_free(&r);
        unsigned e = 1024;
        while (e < (unsigned)n && e < 32768) e <<= 1;
        cap = e;
        if ((inited = ring_init(&r, e)) < 0) return inited;
    }
    long bad = 0;
    unsigned mask = *r.sq_mask, cmask = *r.cq_mask;
    long W = (long)cap < 4096 ? (long)cap : 4096L;   /* in-flight window */
    for (long base_i = 0; base_i < n; base_i += W) {
        long chunk = (n - base_i < W) ? (n - base_i) : W;
        unsigned tail = *r.sq_tail;
        for (long i = 0; i < chunk; i++) {
            unsigned slot = (tail + i) & mask;
            struct io_uring_sqe *sqe = &r.sqes[slot];
            memset(sqe, 0, sizeof(*sqe));
            sqe->opcode = IORING_OP_READ;
            if (async) sqe->flags |= IOSQE_ASYNC;
            sqe->fd = fds[base_i + i];
            sqe->off = (__u64)offs[base_i + i];
            sqe->addr = (__u64)(base + (base_i + i) * row_bytes);
            sqe->len = (__u32)row_bytes;
            sqe->user_data = (__u64)(base_i + i);
            r.sq_array[slot] = slot;
        }
        __atomic_store_n(r.sq_tail, tail + chunk, __ATOMIC_RELEASE);
        long submitted = 0;
        while (submitted < chunk) {
            long ret = syscall(__NR_io_uring_enter, r.ring_fd,
                               (unsigned)(chunk - submitted),
                               (unsigned)(chunk - submitted),
                               IORING_ENTER_GETEVENTS, NULL, (size_t)0);
            if (ret < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                return -errno;
            }
            submitted += ret > 0 ? ret : 0;
        }
        long got = 0;
        while (got < chunk) {
            while (*r.cq_head != *r.cq_tail) {
                struct io_uring_cqe *cqe = &r.cqes[*r.cq_head & cmask];
                if (cqe->res != (int)row_bytes) bad++;
                (*r.cq_head)++;
                got++;
            }
            if (got < chunk) {
                long ret = syscall(__NR_io_uring_enter, r.ring_fd, 0,
                                   (unsigned)(chunk - got),
                                   IORING_ENTER_GETEVENTS, NULL, (size_t)0);
                if (ret < 0 && errno != EINTR && errno != EAGAIN)
                    return -errno;
            }
        }
    }
    return bad;
}
