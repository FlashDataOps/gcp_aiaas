pipeline {
  agent none // No default agent; each stage will define its own
  stages {
    stage('Code Test Stage') {
      agent { 
        docker { 
          image 'python:3.11' // Python Docker image
          args '-v /var/run/docker.sock:/var/run/docker.sock --user root'
        } 
      }
      steps {
        sh "python --version" // Run Python commands
      }
    }
    stage('Docker Build Stage') {
    agent {
      docker {
        image 'google/cloud-sdk:latest'
        // run sudo chmod 666 /var/run/docker.sock in host to allow accessing container's host Docker server
        args '-v /var/run/docker.sock:/var/run/docker.sock --user root'
      }
    }
      environment {
          CLOUDSDK_CONFIG = "${env.WORKSPACE}/gcloud-config"  // Set a writable directory for gcloud
          CLOUDSDK_CORE_PROJECT='single-cirrus-435319-f1'
          GCLOUD_CREDS=credentials('gcloud-creds')
          CLOUDSDK_PYTHON_LOG_FILE = "${env.WORKSPACE}/gcloud-config/logs" // Set writable log path
          DOCKER_CONFIG = "${env.WORKSPACE}/docker-config" // Explicitly define DOCKER_CONFIG
          IMAGE_TAG = "${env.BUILD_NUMBER}"
      }
      steps {
          sh '''
            gcloud version
            gcloud auth activate-service-account --key-file="$GCLOUD_CREDS"
            gcloud config set project $CLOUDSDK_CORE_PROJECT
            # Create Docker config directory
            mkdir -p $DOCKER_CONFIG
            gcloud auth configure-docker --quiet
            # Build and push the Docker image to cloud registry
            docker build --platform linux/amd64 -t gcr.io/$CLOUDSDK_CORE_PROJECT/hello-world:$IMAGE_TAG .
            docker push gcr.io/$CLOUDSDK_CORE_PROJECT/hello-world:$IMAGE_TAG
          '''
      }
    }
    stage('Terraform Deployment Stage') {
      agent {
        docker {
            image 'hashicorp/terraform:light'
            args '-i --entrypoint='
            args '-v /var/run/docker.sock:/var/run/docker.sock --user root'
        }
      }
      environment {
        TF_VAR_project_id = 'single-cirrus-435319-f1'
        TF_VAR_region = 'us-central1'
        TF_VAR_gcloud_creds=credentials('gcloud-creds')
        TF_VAR_image_tag = "${env.BUILD_NUMBER}"
      }
      steps {
        sh '''
          terraform init
          terraform apply -auto-approve
        '''
      }
    }
  }
}
