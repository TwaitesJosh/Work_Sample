import backend.job as job
import json
import numpy as np

CONFIGS = json.load(open('backend/CONFIG.json'))

def initialise_job_list(num_of_jobs):
    job_list = [job.Job(i) for i in np.arange(num_of_jobs)]
    return job_list

job_list = initialise_job_list(num_of_jobs=job.num_of_jobs)

class JobSearch():

    def __init__(self, job_history, is_green, is_digital):

        self.jobs = [job.Job(x) for x in job_history]

        self.is_green = is_green
        self.is_digital = is_digital

        self.job_matches = self.job_similarity()
        self.digital_matches = np.array([x.job2 for x in self.job_matches if x.job2.is_digital])
        self.green_matches = np.array([x.job2 for x in self.job_matches if x.job2.is_green])
        self.green_and_digital_matches = np.array([x.job2 for x in self.job_matches if x.job2.is_green and x.job2.is_digital])


    def get_best_matches(self, numb_of_matches):
        if self.is_green and self.is_digital:
            return self.green_and_digital_matches[:numb_of_matches]
        
        if self.is_green:
            return self.green_matches[:numb_of_matches]
        if self.is_digital:
            return self.digital_matches[:numb_of_matches]
        
        return self.job_matches[:numb_of_matches]


    def job_similarity(self):
        compared_jobs = np.array([job.compareJobs(self.jobs, job_i) for job_i in job_list])
        similarites = np.vstack([x.comparison for x in compared_jobs])
        
        # TODO: Check this is accurate
        weights = np.array([
            CONFIGS["WEIGHTS"]["ESSENTIAL SKILLS"],
            CONFIGS["WEIGHTS"]["OPTIONAL SKILLS"],
            CONFIGS["WEIGHTS"]["WORK CONTEXT"],
            CONFIGS["WEIGHTS"]["WORK AREA"]
            ])

        weighted_similarites = weights * similarites
        sorted_job = np.argsort(np.nanmean(weighted_similarites,1))
        return compared_jobs[sorted_job]

