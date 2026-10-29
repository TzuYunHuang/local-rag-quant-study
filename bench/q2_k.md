| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen2 3B Q2_K - Medium         |   1.28 GiB |     3.40 B | MTL,BLAS   |       4 |           pp512 |        431.98 ± 0.38 |
| qwen2 3B Q2_K - Medium         |   1.28 GiB |     3.40 B | MTL,BLAS   |       4 |           tg128 |         46.33 ± 0.08 |

build: bfe65e1df (11591)
