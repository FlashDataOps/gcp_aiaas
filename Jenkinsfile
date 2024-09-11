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
    stage('Terraform Stage') {
      agent { 
        docker { 
          image 'hashicorp/terraform:latest' // Terraform Docker image
        } 
      }
      steps {
        sh "terraform --version" // Run Terraform commands
      }
    }
  }
}
