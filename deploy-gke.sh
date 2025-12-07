# deploy-gke.sh
# Complete GKE Deployment Script for LedgerX

#!/bin/bash

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================
PROJECT_ID="ledgerx-mlops"
CLUSTER_NAME="ledgerx-cluster"
REGION="us-central1"
NAMESPACE="ledgerx"
IMAGE_NAME="gcr.io/${PROJECT_ID}/ledgerx-api"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "LedgerX GKE Deployment Script"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Prerequisites Check
# ============================================================
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}✗ gcloud CLI not found. Please install: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi
echo -e "${GREEN}✓ gcloud CLI found${NC}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found. Installing...${NC}"
    gcloud components install kubectl
fi
echo -e "${GREEN}✓ kubectl found${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Please install Docker${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Set project
echo -e "${YELLOW}Setting GCP project to ${PROJECT_ID}...${NC}"
gcloud config set project ${PROJECT_ID}
echo -e "${GREEN}✓ Project set${NC}"

echo ""

# ============================================================
# Step 2: Create GKE Cluster (if doesn't exist)
# ============================================================
echo -e "${YELLOW}Step 2: Checking GKE cluster...${NC}"

if gcloud container clusters describe ${CLUSTER_NAME} --region=${REGION} &> /dev/null; then
    echo -e "${GREEN}✓ Cluster ${CLUSTER_NAME} already exists${NC}"
else
    echo -e "${YELLOW}Creating GKE cluster ${CLUSTER_NAME}...${NC}"
    gcloud container clusters create ${CLUSTER_NAME} \
        --region ${REGION} \
        --num-nodes 2 \
        --machine-type n1-standard-2 \
        --enable-autoscaling \
        --min-nodes 1 \
        --max-nodes 4 \
        --enable-autorepair \
        --enable-autoupgrade \
        --enable-stackdriver-kubernetes \
        --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
        --workload-pool=${PROJECT_ID}.svc.id.goog \
        --enable-shielded-nodes
    
    echo -e "${GREEN}✓ Cluster created${NC}"
fi

# Get cluster credentials
echo -e "${YELLOW}Getting cluster credentials...${NC}"
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION}
echo -e "${GREEN}✓ Credentials configured${NC}"

echo ""

# ============================================================
# Step 3: Build and Push Docker Image
# ============================================================
echo -e "${YELLOW}Step 3: Building and pushing Docker image...${NC}"

# Configure Docker for GCR
gcloud auth configure-docker gcr.io

# Build image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile.api .

# Tag with version
IMAGE_TAG_VERSION="v$(date +%Y%m%d-%H%M%S)"
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:${IMAGE_TAG_VERSION}

# Push images
echo -e "${YELLOW}Pushing images to GCR...${NC}"
docker push ${IMAGE_NAME}:${IMAGE_TAG}
docker push ${IMAGE_NAME}:${IMAGE_TAG_VERSION}

echo -e "${GREEN}✓ Images pushed:${NC}"
echo "   - ${IMAGE_NAME}:${IMAGE_TAG}"
echo "   - ${IMAGE_NAME}:${IMAGE_TAG_VERSION}"

echo ""

# ============================================================
# Step 4: Create Kubernetes Resources
# ============================================================
echo -e "${YELLOW}Step 4: Deploying to Kubernetes...${NC}"

# Create namespace
echo -e "${YELLOW}Creating namespace...${NC}"
kubectl apply -f k8s/namespace.yaml
echo -e "${GREEN}✓ Namespace created${NC}"

# Create ConfigMap
echo -e "${YELLOW}Creating ConfigMap...${NC}"
kubectl apply -f k8s/configmap.yaml
echo -e "${GREEN}✓ ConfigMap created${NC}"

# Create Secrets (check if exists first)
echo -e "${YELLOW}Creating Secrets...${NC}"
if kubectl get secret ledgerx-secrets -n ${NAMESPACE} &> /dev/null; then
    echo -e "${YELLOW}⚠ Secrets already exist, skipping...${NC}"
else
    echo -e "${RED}⚠ Please create secrets manually:${NC}"
    echo "   kubectl create secret generic ledgerx-secrets \\"
    echo "     --from-literal=DB_USER='your-user' \\"
    echo "     --from-literal=DB_PASSWORD='your-password' \\"
    echo "     --from-literal=JWT_SECRET_KEY='your-jwt-secret' \\"
    echo "     -n ${NAMESPACE}"
    echo ""
    read -p "Press Enter after creating secrets..."
fi

# Deploy application
echo -e "${YELLOW}Deploying application...${NC}"
kubectl apply -f k8s/deployment.yaml
echo -e "${GREEN}✓ Deployment created${NC}"

# Create service
echo -e "${YELLOW}Creating service...${NC}"
kubectl apply -f k8s/service.yaml
echo -e "${GREEN}✓ Service created${NC}"

# Deploy ingress
echo -e "${YELLOW}Deploying ingress...${NC}"
kubectl apply -f k8s/ingress.yaml
echo -e "${GREEN}✓ Ingress created${NC}"

# Deploy HPA
echo -e "${YELLOW}Deploying autoscaler...${NC}"
kubectl apply -f k8s/hpa.yaml
echo -e "${GREEN}✓ HPA created${NC}"

echo ""

# ============================================================
# Step 5: Wait for Deployment
# ============================================================
echo -e "${YELLOW}Step 5: Waiting for deployment to be ready...${NC}"

kubectl wait --for=condition=available --timeout=300s \
    deployment/ledgerx-api -n ${NAMESPACE}

echo -e "${GREEN}✓ Deployment is ready${NC}"

echo ""

# ============================================================
# Step 6: Get Access Information
# ============================================================
echo -e "${YELLOW}Step 6: Getting access information...${NC}"

# Get external IP (may take a few minutes)
echo -e "${YELLOW}Waiting for external IP (this may take 5-10 minutes)...${NC}"
EXTERNAL_IP=""
while [ -z $EXTERNAL_IP ]; do
    EXTERNAL_IP=$(kubectl get ingress ledgerx-ingress -n ${NAMESPACE} \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    [ -z "$EXTERNAL_IP" ] && sleep 10
done

echo ""
echo "============================================================"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo "============================================================"
echo ""
echo "📊 Deployment Information:"
echo "   • Cluster: ${CLUSTER_NAME}"
echo "   • Region: ${REGION}"
echo "   • Namespace: ${NAMESPACE}"
echo "   • Image: ${IMAGE_NAME}:${IMAGE_TAG_VERSION}"
echo ""
echo "🌐 Access Information:"
echo "   • External IP: ${EXTERNAL_IP}"
echo "   • Health Check: http://${EXTERNAL_IP}/health"
echo "   • API Docs: http://${EXTERNAL_IP}/docs"
echo "   • Metrics: http://${EXTERNAL_IP}/metrics"
echo ""
echo "📝 Useful Commands:"
echo "   • Check pods: kubectl get pods -n ${NAMESPACE}"
echo "   • Check logs: kubectl logs -f <pod-name> -n ${NAMESPACE}"
echo "   • Check service: kubectl get svc -n ${NAMESPACE}"
echo "   • Check ingress: kubectl get ingress -n ${NAMESPACE}"
echo "   • Check HPA: kubectl get hpa -n ${NAMESPACE}"
echo "   • Scale manually: kubectl scale deployment ledgerx-api --replicas=5 -n ${NAMESPACE}"
echo ""
echo "🔍 Monitor deployment:"
echo "   kubectl get pods -n ${NAMESPACE} -w"
echo ""
echo "🧪 Test API:"
echo "   curl http://${EXTERNAL_IP}/health"
echo ""

# ============================================================
# Optional: Open dashboard
# ============================================================
read -p "Open Kubernetes dashboard? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl proxy &
    echo "Dashboard available at: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/"
fi

echo ""
echo -e "${GREEN}Deployment script completed successfully!${NC}"