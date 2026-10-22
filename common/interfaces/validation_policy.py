"""Report requirements shared by pool registration and local evaluation."""
import os


def required_reports():
    reports = ['utility', 'privacy']
    if os.getenv('FIDELITY_ENABLED', 'false').lower() == 'true':
        reports.append('fidelity')
    return reports
