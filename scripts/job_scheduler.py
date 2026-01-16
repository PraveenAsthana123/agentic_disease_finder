#!/usr/bin/env python3
"""
Job Scheduler - Tracks training jobs per disease
Shows: Data processing status, Model training, Accuracy
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
import threading

BASE_DIR = Path('/media/praveen/Asthana3/rajveer/agenticfinder')
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'saved_models'
STATUS_FILE = BASE_DIR / 'results' / 'job_status.json'


@dataclass
class DiseaseJob:
    name: str
    status: str = "pending"  # pending, loading, processing, training, completed, failed
    data_samples: int = 0
    model_type: str = "VotingClassifier"
    accuracy: float = 0.0
    std: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None
    model_path: Optional[str] = None


class JobScheduler:
    def __init__(self):
        self.jobs = {
            'epilepsy': DiseaseJob('Epilepsy'),
            'schizophrenia': DiseaseJob('Schizophrenia'),
            'depression': DiseaseJob('Depression'),
            'stress': DiseaseJob('Stress'),
            'autism': DiseaseJob('Autism'),
            'parkinson': DiseaseJob('Parkinson'),
            'dementia': DiseaseJob('Dementia'),
        }
        self.load_status()

    def load_status(self):
        """Load previous status if exists"""
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE) as f:
                    data = json.load(f)
                for name, job_data in data.items():
                    if name in self.jobs:
                        for k, v in job_data.items():
                            setattr(self.jobs[name], k, v)
            except:
                pass

    def save_status(self):
        """Save current status"""
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.jobs.items()}, f, indent=2)

    def update_job(self, name: str, **kwargs):
        """Update job status"""
        if name in self.jobs:
            for k, v in kwargs.items():
                setattr(self.jobs[name], k, v)
            self.save_status()

    def print_status(self):
        """Print current status table"""
        print("\n" + "="*90)
        print(f"{'DISEASE':<15} {'STATUS':<12} {'SAMPLES':<10} {'MODEL':<20} {'ACCURACY':<15} {'TIME'}")
        print("="*90)

        for name, job in self.jobs.items():
            acc_str = f"{job.accuracy:.1f}% (+/-{job.std:.1f})" if job.accuracy > 0 else "-"
            time_str = job.start_time or "-"
            samples_str = str(job.data_samples) if job.data_samples > 0 else "-"

            status_color = {
                'pending': '\033[90m',
                'loading': '\033[93m',
                'processing': '\033[93m',
                'training': '\033[94m',
                'completed': '\033[92m',
                'failed': '\033[91m',
            }.get(job.status, '')

            print(f"{job.name:<15} {status_color}{job.status:<12}\033[0m {samples_str:<10} {job.model_type:<20} {acc_str:<15} {time_str}")

        print("="*90)

    def run_job(self, disease: str):
        """Run training for a single disease"""
        job = self.jobs.get(disease)
        if not job:
            return

        print(f"\n{'#'*60}")
        print(f"# STARTING: {job.name.upper()}")
        print(f"{'#'*60}")

        job.start_time = datetime.now().strftime('%H:%M:%S')
        job.status = 'loading'
        self.save_status()

        try:
            # Import training module
            sys.path.insert(0, str(BASE_DIR / 'scripts'))
            from train_sequential import (
                load_epilepsy, load_schizophrenia, load_depression,
                load_stress, load_autism, load_parkinson, load_dementia,
                train_and_test
            )

            loaders = {
                'epilepsy': load_epilepsy,
                'schizophrenia': load_schizophrenia,
                'depression': load_depression,
                'stress': load_stress,
                'autism': load_autism,
                'parkinson': load_parkinson,
                'dementia': load_dementia,
            }

            # Load data
            data = loaders[disease]()
            job.data_samples = len(data)
            job.status = 'processing'
            self.save_status()

            if len(data) < 10:
                job.status = 'failed'
                job.error = f"Insufficient data: {len(data)} samples"
                self.save_status()
                return

            # Train
            job.status = 'training'
            self.save_status()

            result = train_and_test(data, job.name)

            if result:
                job.accuracy = result['accuracy']
                job.std = result['std']
                job.model_path = str(MODELS_DIR / f"{disease}_model.joblib")
                job.status = 'completed'
            else:
                job.status = 'failed'
                job.error = "Training failed"

        except Exception as e:
            job.status = 'failed'
            job.error = str(e)

        job.end_time = datetime.now().strftime('%H:%M:%S')
        self.save_status()
        self.print_status()

    def run_all(self):
        """Run all jobs sequentially"""
        print("\n" + "="*60)
        print("DISEASE CLASSIFICATION - JOB SCHEDULER")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        self.print_status()

        for disease in self.jobs.keys():
            if self.jobs[disease].status != 'completed':
                self.run_job(disease)
                import gc
                gc.collect()

        print("\n" + "="*60)
        print("ALL JOBS COMPLETED")
        print("="*60)
        self.print_status()

        # Summary
        completed = [j for j in self.jobs.values() if j.status == 'completed']
        achieved = [j for j in completed if j.accuracy >= 90]
        print(f"\nCompleted: {len(completed)}/{len(self.jobs)}")
        print(f"Achieved 90%+: {len(achieved)}/{len(completed)}")


def main():
    scheduler = JobScheduler()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == 'status':
            scheduler.print_status()
        elif cmd == 'reset':
            for job in scheduler.jobs.values():
                job.status = 'pending'
                job.accuracy = 0
                job.std = 0
                job.data_samples = 0
            scheduler.save_status()
            print("All jobs reset to pending")
        elif cmd in scheduler.jobs:
            scheduler.run_job(cmd)
        else:
            print(f"Usage: {sys.argv[0]} [status|reset|<disease_name>|all]")
            print(f"Diseases: {list(scheduler.jobs.keys())}")
    else:
        scheduler.run_all()


if __name__ == "__main__":
    main()
