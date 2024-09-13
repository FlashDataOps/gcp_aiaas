pipeline {
  agent none
  environment {
    CLOUDSDK_CORE_PROJECT='single-cirrus-435319-f1'
    GCLOUD_CREDS=credentials('gcloud-creds')
  }
  stages {
    stage('Python Stage') {
      agent { 
        docker { 
          image 'python:latest'
        } 
      }
      steps {
        node {
          sh "python --version"
        }
      }
    }
    stage('GCloud Stage') {
      agent {
        docker {
          image 'google/cloud-sdk:latest'
        }
      }
      steps {
        node {
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
        node {
          sh "terraform --version"
        }
      }
    }
  }
}