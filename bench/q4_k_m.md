| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen2 3B Q4_K - Medium         |   1.95 GiB |     3.40 B | MTL,BLAS   |       4 |           pp512 |        429.60 ± 0.15 |
| qwen2 3B Q4_K - Medium         |   1.95 GiB |     3.40 B | MTL,BLAS   |       4 |           tg128 |         41.42 ± 0.38 |

build: bfe65e1df (11591)
