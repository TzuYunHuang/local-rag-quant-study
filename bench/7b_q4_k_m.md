| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | MTL,BLAS   |       4 |           pp512 |        186.97 ± 0.47 |
| qwen2 7B Q4_K - Medium         |   4.36 GiB |     7.62 B | MTL,BLAS   |       4 |           tg128 |         19.22 ± 0.61 |

build: bfe65e1df (11591)
