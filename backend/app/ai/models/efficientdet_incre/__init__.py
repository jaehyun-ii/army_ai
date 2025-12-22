"""
Custom EfficientDet models with incremental learning support.

This module provides custom implementations of EfficientDet models
that support incremental learning for continuously adding new object classes.
"""
from .efficientdet_incre import EfficientDetIncre
from .efficientdet_head_incre import EfficientDetSepBNHeadIncre

__all__ = ['EfficientDetIncre', 'EfficientDetSepBNHeadIncre']
