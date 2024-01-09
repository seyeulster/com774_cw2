from locust import HttpUser, task

class InferenceUser(HttpUser):
    @task
    def make_inference(self):
        self.client.post("/invocations", json={
            "inputs": [[0.04,0.09,0.14,0.12,0.11,0.1,0.08,0.13,0.13,0.08,0.09,0.1,0.11,0.11,0.08,0.04,0.16,0.13,0.1,0.03,0.12,0.08,0.09,0.12,0.1,0.1,0.08,0.11,0.12,0.1,0,8.4,1.76,2075,293.94,1550,3.29,7.21,4,4.05,8.17,4.05,11.96]]
        })