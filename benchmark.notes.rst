Notes
-----

The benchmarks in `results.csv` were run on my Lenov Carbon X1 which has a
Intel Core i7 3667u 2 cores / 4 threads clocked at 2.0 Ghz / thread

* Sizes:

  :small: 1e4 * 8 = 80000 Bytes =80KB
  :mid: 1e7 * 8 = 80000000 Bytes = 80MB
  :large: 2e8 * 8 = 1600000000 Bytes = 1.4 GB

* Entropy:

  :low: `arange` (very low Kolmogorov complexity)
  :medium: Sinwave + noise
  :high: random numbers

* Storage, io time measures using `dd` with `/dev/null`:

  :ssd: encrypted (LUKS) SSD 230 MB/s write / 350 MB/sd read
  :sd:  SD card  20 MB/sd read/write (simulates slow storage medium)
