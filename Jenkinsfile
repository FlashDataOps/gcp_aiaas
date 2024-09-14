pipeline {
  agent none // No default agent; each stage will define its own
  stages {
    stage('Python Stage') {
      agent { 
        docker { 
          image 'python:latest' // Python Docker image
        } 
      }
      steps {
        sh "python --version" // Run Python commands
        sh "python HelloWorld.py"
      }
    }
    stage('GCloud Stage') {
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
            docker build -t gcr.io/$CLOUDSDK_CORE_PROJECT/hello-world:latest .
            docker push gcr.io/$CLOUDSDK_CORE_PROJECT/hello-world:latest
          '''
      }
    }
    stage('Terraform Stage') {
      agent {
        docker {
            image 'hashicorp/terraform:light'
            args '-i --entrypoint='
        }
      }
      environment {
        TF_VAR_project_id = 'single-cirrus-435319-f1'
        TF_VAR_region = 'us-central1'
        GCLOUD_CREDS=credentials('gcloud-creds')
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
