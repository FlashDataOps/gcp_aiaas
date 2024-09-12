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
      steps {
        sh "gcloud version"
        sh "gcloud compute zones list"
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
