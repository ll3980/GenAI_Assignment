# Verification of the final submission

The application was checked locally on 2026-10-01T00:18:45.212733+00:00. The API uses the actual pretrained `en_core_web_lg` 3.8.0 model. The saved responses in `examples/` were captured from the running HTTP server.

| Check | Result |
| --- | --- |
| Dependency lock check | `uv lock --check --offline` succeeded |
| Unit and API tests | `uv run --offline pytest -q`: 19 passed |
| Independent model comparison | All returned vector values exactly equal the full classroom spaCy pipeline's `nlp("apple").vector` |
| Live HTTP checks | All six checks in `scripts/check_api.py` passed against the local server |
| Failed-server detection | The live check script returned exit status 1 when the server was unavailable |
| Python, Compose and workflow syntax | Checked locally |
| Image model-check command | Succeeded when executed with the local installed dependencies |
| Probability questions | All six solutions were checked against the original assignment; the two PDF pages were visually reviewed |
| Actual Docker image build and run | Not executed in this workspace because a Docker engine is unavailable |
| GitHub commit and Actions run | Must be completed in the student's repository |

The live HTTP checks cover server status, Swagger UI and OpenAPI routes, a complete finite nonzero 300-dimensional vector, GET/POST agreement, documented input errors, and preservation of the original text generator. The unit tests additionally check each generated transition, normalized bigram distributions, terminal words, and exact vector equality with the full spaCy model.

To complete container verification, run the Docker commands in the README or commit the included GitHub workflow and confirm **Docker API checks** succeeds. The workflow builds the actual image and queries the API through the published host port. A local Python test does not establish that a Docker image has successfully built or run.

## Probability results

| Question | Checked result |
| --- | --- |
| 1(a), 1(b) | 0.12; 0.58 |
| 2 | Not independent, since P(A given B) = 0.7 differs from P(A) = 0.5 |
| 3 | 15/19, approximately 0.78947 |
| 4 | 19/117, approximately 16.24% |
| 5(a), 5(b), 5(c) | Expected value 90; variance 25; sample mean 90 |
| 6(a), 6(b) | Approximately 1.846439 bits; 2 bits |
