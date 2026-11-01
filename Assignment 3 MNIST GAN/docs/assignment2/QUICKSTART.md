# Assignment 2 quick start

The current repository combines the original Assignment 1 API, the trained
CIFAR-10 classifier, and the Assignment 3 GAN. Follow the root README to
start the complete application.

The classifier checkpoint is `models/cifar10_cnn.pt`. Serving it does not
require downloading CIFAR-10 or retraining. Open `/docs`, select POST
`/predict`, and upload `examples/cat.png` or another readable image. The
original `/generate` and `/embedding` endpoints are retained.

Start Docker from the repository root:

```bash
docker compose up --build -d --wait --wait-timeout 180
```

Open `http://localhost:8000/docs`. Internal port 80 is mapped to host port
8000. Check `/cnn/health`, classification, text generation, and embeddings
before submitting.

English theory solutions are in `Assignment2_Theory_LiuyangLi.pdf` in this
directory. Include the actual repository URL and commit hash as required.
If code must also be uploaded, preserve its directory structure and include
the trained checkpoint. Existing training/evaluation records are in
`results/assignment2/`. Actual Docker execution still needs confirmation on
a Docker-enabled machine.
