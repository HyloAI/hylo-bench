# code.llm on LIBERO (qwen3.5:35b-a3b via Ollama, vision, zero-shot, max 16 turns)

Model server on this workstation's CPU, shared with other jobs: seconds per turn are mostly queueing.

| task | init state | outcome | robot steps | turns | blocks with errors | s / turn | tokens in / out | wall min |
|---|---|---|---|---|---|---|---|---|
| libero_object / pick up the alphabet soup and place it in the basket | 0 | success | 126 | 1 | 0 | 1681 | 4582 / 773 | 28 |
| libero_object / pick up the alphabet soup and place it in the basket | 1 | success | 165 | 3 | 0 | 1321 | 15715 / 984 | 67 |
| libero_object / pick up the alphabet soup and place it in the basket | 2 | success | 137 | 4 | 1 | 614 | 23414 / 1026 | 41 |
| libero_spatial / pick up the black bowl between the plate and the ramekin and place it on the plate | 0 | failure (280-step budget used up) | 280 | 13 | 4 | 902 | 165740 / 10880 | 197 |
| libero_spatial / (same task) | 1 | failure (moved the other black bowl to the plate, then finish()) | 124 | 7 | 0 | 650 | 54416 / 2712 | 76 |
| libero_spatial / (same task) | 2 | failure (280-step budget used up) | 280 | 9 | 1 | 202 | 88990 / 8314 | 32 |
| libero_goal / put the wine bottle on the rack | 0 | success | 131 | 5 | 0 | 700 | 29948 / 1439 | 59 |

libero_object 3/3, libero_spatial 0/3, libero_goal 1/1 (an extra task, one episode). Init states 1 and 2 of libero_spatial were
first interrupted when the model server was stopped (0 robot steps); those attempts are not counted, and the rows above
are from a fresh run (one model session at a time).
