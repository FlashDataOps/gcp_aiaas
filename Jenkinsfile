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
      }
    }
    stage('GCloud Stage') {
      agent {
        docker {
          image 'google/cloud-sdk:latest'
        }
      }
      environment {
          CLOUDSDK_CONFIG = "${env.WORKSPACE}/gcloud-config"  // Set a writable directory for gcloud
          CLOUDSDK_PYTHON_LOG_FILE = "${env.WORKSPACE}/gcloud-config/logs" // Set writable log path
      }
      steps {
        withCredentials([file(credentialsId: 'gcloud-creds', variable: 'GCLOUD_CREDS')]) {
          sh '''
            gcloud version
            gcloud auth activate-service-account --key-file="$GCLOUD_CREDS"
            gcloud compute zones list
          '''
        }
      }
    }
    stage('Terraform Stage') {
      agent {
        docker {
            image 'hashicorp/terraform:light'
            args '-i --entrypoint='
        }
      }
      steps {
        sh "terraform --version" // Run Terraform commands
      }
    }
  }
}
