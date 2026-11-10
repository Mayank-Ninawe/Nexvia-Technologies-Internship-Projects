# Deployment Status and Verification

## Project deployment summary

The real-time financial fraud detection platform has been deployed and verified across local application execution, Docker-based Kafka streaming, Docker image packaging, and Minikube Kubernetes deployment.

The FastAPI fraud-scoring API was successfully built as a Docker image, loaded into Minikube, deployed through Kubernetes, and verified as healthy.

## Verified local application

The following components were implemented and verified locally:

- Python virtual environment with tested project dependencies.
- PaySim data ingestion and validation workflow.
- Leakage-safe feature engineering.
- XGBoost fraud scoring model.
- FastAPI model-serving API.
- Streamlit analyst dashboard.
- Kafka producer and consumer workflow.
- MLflow experiment tracking.
- Feast feature-store consistency checks.
- Evidently drift monitoring and review trigger.
- Graph-based fraud-ring feasibility analysis.

## Data integrity verification

The PaySim ingestion adapter preserves all source transactions and fraud labels.

```text
Raw rows: 6,362,620
Processed rows: 6,362,620
Raw fraud rows: 8,213
Processed fraud rows: 8,213
Zero-amount fraud rows preserved: 16
```

A preprocessing issue that initially removed 16 zero-amount fraudulent transactions was identified and corrected before modeling.

## Docker verification

Docker Desktop was installed and verified.

```powershell
docker run hello-world
```

Docker Compose was used to run the Kafka broker locally.

```powershell
docker compose -f infra\docker\docker-compose.kafka.yml up -d
```

Kafka producer and consumer workflows were verified by replaying PaySim transaction events, consuming them, calling the FastAPI scoring service, and persisting scored alerts.

## FastAPI container packaging

The repository includes the following deployment configuration:

```text
infra\docker\Dockerfile.api
requirements-api.txt
infra\docker\docker-compose.yml
.dockerignore
infra\k8s\fraud-api-deployment.yaml
infra\k8s\fraud-api-service.yaml
```

The API image uses a minimal runtime dependency file and includes only the API and source code required for serving.

The Docker image was successfully built:

```powershell
docker compose -f infra\docker\docker-compose.yml build --no-cache fraud-api
```

Build result:

```text
✔ Image realtime-financial-fraud-detection-fraud-api Built
```

## Minikube installation

Minikube v1.39.0 was installed successfully.

```powershell
minikube version
```

Output:

```text
minikube version: v1.39.0
commit: 7a9f6a841470a207de8cf4bafcccee0969d8ba10
```

Minikube state was configured on the D: drive to avoid filling the system drive:

```powershell
$env:MINIKUBE_HOME = "D:\minikube-home"
```

## Minikube cluster startup

The local Kubernetes cluster was started using the Docker driver:

```powershell
minikube start --driver=docker
```

Successful startup output:

```text
Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default
```

Cluster verification:

```powershell
minikube status
kubectl get nodes
kubectl get pods -A
```

Verified status:

```text
minikube
type: Control Plane
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured
```

Verified node:

```text
NAME       STATUS   ROLES           AGE   VERSION
minikube   Ready    control-plane   10m   v1.37.0
```

## Kubernetes API deployment

The API image was loaded into Minikube:

```powershell
minikube image load realtime-financial-fraud-detection-fraud-api:latest
```

The Kubernetes resources were applied:

```powershell
kubectl apply -f infra\k8s\fraud-api-deployment.yaml
kubectl apply -f infra\k8s\fraud-api-service.yaml
```

Created resources:

```text
deployment.apps/fraud-api created
service/fraud-api created
```

## API pod verification

The API pod reached a healthy running state:

```powershell
kubectl get pods -l app=fraud-api
```

Output:

```text
NAME                         READY   STATUS    RESTARTS   AGE
fraud-api-77cf7f7cdb-shcqx   1/1     Running   0          102s
```

Pod details confirmed:

```text
Image: realtime-financial-fraud-detection-fraud-api:latest
State: Running
Port: 8000/TCP
Readiness: http-get http://:8000/health
Liveness: http-get http://:8000/health
```

## API health verification

The API logs confirmed successful startup:

```text
INFO: Started server process[1]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: ... "GET /health HTTP/1.1" 200 OK
```

The service was exposed locally through Minikube:

```powershell
minikube service fraud-api --url
```

Health check:

```powershell
curl.exe http://127.0.0.1:63861/health
```

Response:

```json
{
  "status": "healthy",
  "service": "fraud-detection-api"
}
```

## Kubernetes service configuration

The API uses a `ClusterIP` Service for stable in-cluster communication:

```text
NAME        TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
fraud-api   ClusterIP   10.106.207.61   <none>        80/TCP    3s
```

For local development, Minikube provides a temporary local proxy URL:

```text
http://127.0.0.1:63861
```

## Local demonstration commands

### Start Minikube and API

```powershell
cd D:\Nexvia-Internship\realtime-financial-fraud-detection
$env:MINIKUBE_HOME = "D:\minikube-home"
minikube start --driver=docker
kubectl get pods -l app=fraud-api
minikube service fraud-api --url
```

### Run Kafka

```powershell
docker compose -f infra\docker\docker-compose.kafka.yml up -d
```

### Run local API and dashboard

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --app-dir . --port 8000
streamlit run backend\dashboard\app.py
```

## Cleanup commands

```powershell
kubectl delete -f infra\k8s\fraud-api-service.yaml
kubectl delete -f infra\k8s\fraud-api-deployment.yaml
minikube stop
docker compose -f infra\docker\docker-compose.kafka.yml down
```

## Known limitations

- Kafka currently runs through Docker Compose rather than inside the Minikube cluster.
- The Kubernetes Service is `ClusterIP`; local access uses Minikube’s temporary service proxy.
- PaySim does not provide real device, IP, merchant, or location identifiers; enrichment fields used for graph analysis are simulated deterministically from account identifiers.
- The project does not include production-grade secrets management, autoscaling, or a cloud Kubernetes cluster.

## Final deployment status

| Component | Status |
|---|---|
| Docker Desktop | Verified |
| Kafka in Docker Compose | Verified |
| FastAPI Docker image | Built successfully |
| Minikube installation | Complete |
| Minikube cluster | Running and Ready |
| Kubernetes Deployment | Created |
| Kubernetes Service | Created |
| API pod | `1/1 Running` |
| Readiness probe | Passing |
| `/health` endpoint | HTTP 200 |
| Kubernetes API deployment | **Verified complete** |
| React Command Center (`frontend/`) | **Verified (`dist/` built, browser verified)** |
| High-Throughput Streaming Persistence | **Verified (Append-only JSONL $O(1)$ writes)** |
| Automated Pytest Test Suite | **Verified (27/27 tests passed)** |