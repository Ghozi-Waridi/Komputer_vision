"""
streamlit_dashboard package
Minimal module untuk prediction-only interface
"""

from .pages import page_prediction_upload, page_prediction_dataset

__all__ = [
    'page_prediction_upload',
    'page_prediction_dataset'
]

__version__ = '1.0'
