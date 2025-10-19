"""
Tests for node registration and ComfyUI integration.

Verifies that all nodes are properly registered and have correct metadata.
"""

import pytest
import sys
from pathlib import Path

# Import the NODE mappings by reconstructing them from the actual node modules
# instead of importing from __init__.py which has relative imports


class TestNodeRegistration:
    """Test that nodes are properly registered in __init__.py."""

    def test_node_class_mappings_exists(self, node_mappings):
        """Test NODE_CLASS_MAPPINGS is defined."""
        assert "NODE_CLASS_MAPPINGS" in node_mappings
        assert isinstance(node_mappings["NODE_CLASS_MAPPINGS"], dict)

    def test_node_display_name_mappings_exists(self, node_mappings):
        """Test NODE_DISPLAY_NAME_MAPPINGS is defined."""
        assert "NODE_DISPLAY_NAME_MAPPINGS" in node_mappings
        assert isinstance(node_mappings["NODE_DISPLAY_NAME_MAPPINGS"], dict)

    def test_all_nodes_have_display_names(self, node_mappings):
        """Test that all registered nodes have display names."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]
        display_mappings = node_mappings["NODE_DISPLAY_NAME_MAPPINGS"]

        for node_key in class_mappings.keys():
            assert node_key in display_mappings, f"Node {node_key} missing display name"

    def test_legacy_node_registered(self, node_mappings):
        """Test that legacy node is registered (if available)."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]
        display_mappings = node_mappings["NODE_DISPLAY_NAME_MAPPINGS"]

        # Legacy node might not be available in test environment
        if "TLBVFI_VFI_TF32" in class_mappings:
            assert "TLBVFI_VFI_TF32" in display_mappings

    def test_v2_nodes_registered(self, node_mappings):
        """Test that V2 production nodes are registered."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        v2_nodes = [
            "TLBVFI_Interpolator_V2",
            "TLBVFI_BatchInterpolator_V2",
        ]

        for node_key in v2_nodes:
            assert node_key in class_mappings, f"{node_key} not registered"

    def test_chunk_workflow_nodes_registered(self, node_mappings):
        """Test that chunk workflow nodes are registered."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        # Core chunk nodes that should always be available
        core_chunk_nodes = [
            "TLBVFI_FramePairSlicer",
            "TLBVFI_ChunkProcessor",
        ]

        for node_key in core_chunk_nodes:
            assert node_key in class_mappings, f"{node_key} not registered"

        # Optional chunk nodes
        optional_chunk_nodes = [
            "TLBVFI_Interpolator",
            "TLBVFI_ChunkVideoSaver",
            "TLBVFI_VideoConcatenator",
        ]

        # Just verify they're in the mappings if available
        for node_key in optional_chunk_nodes:
            if node_key in class_mappings:
                assert class_mappings[node_key] is not None

    def test_utility_nodes_registered(self, node_mappings):
        """Test that utility nodes are registered."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        # ClearModelCache should always be available
        assert "TLBVFI_ClearModelCache" in class_mappings

        # FrameFromBatch is optional
        if "TLBVFI_FrameFromBatch" in class_mappings:
            assert class_mappings["TLBVFI_FrameFromBatch"] is not None


class TestNodeClasses:
    """Test that registered node classes are valid."""

    def test_all_nodes_importable(self, node_mappings):
        """Test that all registered node classes can be imported."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        for node_key, node_class in class_mappings.items():
            assert node_class is not None, f"Node {node_key} is None"
            assert callable(node_class), f"Node {node_key} is not callable"

    def test_nodes_have_input_types(self, node_mappings):
        """Test that all nodes have INPUT_TYPES class method."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        for node_key, node_class in class_mappings.items():
            assert hasattr(node_class, 'INPUT_TYPES'), f"{node_key} missing INPUT_TYPES"
            assert callable(node_class.INPUT_TYPES), f"{node_key}.INPUT_TYPES not callable"

    def test_nodes_have_return_types(self, node_mappings):
        """Test that all nodes have RETURN_TYPES defined."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        for node_key, node_class in class_mappings.items():
            # Skip model cache manager as it might not have RETURN_TYPES
            if "ClearModelCache" in node_key:
                continue

            assert hasattr(node_class, 'RETURN_TYPES'), f"{node_key} missing RETURN_TYPES"

    def test_nodes_have_function(self, node_mappings):
        """Test that all nodes have FUNCTION defined."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        for node_key, node_class in class_mappings.items():
            assert hasattr(node_class, 'FUNCTION'), f"{node_key} missing FUNCTION"
            assert isinstance(node_class.FUNCTION, str), f"{node_key}.FUNCTION not a string"


class TestNodeInputTypes:
    """Test INPUT_TYPES structure for all nodes."""

    def test_input_types_returns_dict(self, node_mappings):
        """Test that INPUT_TYPES returns a dictionary."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        for node_key, node_class in class_mappings.items():
            input_types = node_class.INPUT_TYPES()
            assert isinstance(input_types, dict), f"{node_key}.INPUT_TYPES() not a dict"

    def test_input_types_has_required_or_optional(self, node_mappings):
        """Test that INPUT_TYPES has 'required' or 'optional' key."""
        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]

        for node_key, node_class in class_mappings.items():
            input_types = node_class.INPUT_TYPES()

            has_required = "required" in input_types
            has_optional = "optional" in input_types

            assert has_required or has_optional, (
                f"{node_key}.INPUT_TYPES() must have 'required' or 'optional'"
            )


class TestFramePairSlicerNode:
    """Specific tests for FramePairSlicer node registration."""

    def test_frame_pair_slicer_registered(self, node_mappings):
        """Test FramePairSlicer is properly registered."""
        from nodes.frame_pair_slicer import FramePairSlicer

        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]
        assert "TLBVFI_FramePairSlicer" in class_mappings
        assert class_mappings["TLBVFI_FramePairSlicer"] == FramePairSlicer

    def test_frame_pair_slicer_has_correct_inputs(self):
        """Test FramePairSlicer has correct input structure."""
        from nodes.frame_pair_slicer import FramePairSlicer

        input_types = FramePairSlicer.INPUT_TYPES()

        assert "required" in input_types
        assert "images" in input_types["required"]
        assert input_types["required"]["images"] == ("IMAGE",)

        assert "optional" in input_types
        assert "pair_index" in input_types["optional"]

    def test_frame_pair_slicer_has_correct_outputs(self):
        """Test FramePairSlicer has correct output structure."""
        from nodes.frame_pair_slicer import FramePairSlicer

        assert FramePairSlicer.RETURN_TYPES == ("IMAGE", "INT", "INT", "BOOLEAN")
        assert FramePairSlicer.RETURN_NAMES == (
            "frame_pair", "pair_index", "total_pairs", "is_last_pair"
        )


class TestInterpolatorV2Node:
    """Specific tests for TLBVFI_Interpolator_V2 node registration."""

    def test_interpolator_v2_registered(self, node_mappings):
        """Test Interpolator V2 is properly registered."""
        from nodes.tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2

        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]
        assert "TLBVFI_Interpolator_V2" in class_mappings
        assert class_mappings["TLBVFI_Interpolator_V2"] == TLBVFI_Interpolator_V2

    def test_interpolator_v2_has_frame_inputs(self):
        """Test Interpolator V2 has frame inputs."""
        from nodes.tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2

        input_types = TLBVFI_Interpolator_V2.INPUT_TYPES()

        assert "required" in input_types
        required = input_types["required"]

        assert "prev_frame" in required
        assert "next_frame" in required
        assert "model_name" in required

    def test_interpolator_v2_has_optional_params(self):
        """Test Interpolator V2 has optional parameters."""
        from nodes.tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2

        input_types = TLBVFI_Interpolator_V2.INPUT_TYPES()

        assert "optional" in input_types
        optional = input_types["optional"]

        expected_params = [
            "times_to_interpolate",
            "enable_tf32",
            "sample_steps",
            "flow_scale",
            "cpu_offload",
            "gpu_id",
        ]

        for param in expected_params:
            assert param in optional, f"Missing optional parameter: {param}"


class TestChunkProcessorNode:
    """Specific tests for ChunkProcessor node registration."""

    def test_chunk_processor_registered(self, node_mappings):
        """Test ChunkProcessor is properly registered."""
        from nodes.chunk_processor import TLBVFI_ChunkProcessor

        class_mappings = node_mappings["NODE_CLASS_MAPPINGS"]
        assert "TLBVFI_ChunkProcessor" in class_mappings
        assert class_mappings["TLBVFI_ChunkProcessor"] == TLBVFI_ChunkProcessor

    def test_chunk_processor_has_images_input(self):
        """Test ChunkProcessor has images input."""
        from nodes.chunk_processor import TLBVFI_ChunkProcessor

        input_types = TLBVFI_ChunkProcessor.INPUT_TYPES()

        assert "required" in input_types
        assert "images" in input_types["required"]


class TestNodeCategories:
    """Test that nodes have appropriate categories."""

    def test_frame_pair_slicer_category(self):
        """Test FramePairSlicer is in chunk category."""
        from nodes.frame_pair_slicer import FramePairSlicer

        assert hasattr(FramePairSlicer, 'CATEGORY')
        assert "chunk" in FramePairSlicer.CATEGORY.lower()

    def test_interpolator_v2_category(self):
        """Test Interpolator V2 has category defined."""
        from nodes.tlbvfi_interpolator_v2 import TLBVFI_Interpolator_V2

        assert hasattr(TLBVFI_Interpolator_V2, 'CATEGORY')
        assert "frame_interpolation" in TLBVFI_Interpolator_V2.CATEGORY.lower()


class TestNodeDescriptions:
    """Test that critical nodes have descriptions."""

    def test_frame_pair_slicer_has_description(self):
        """Test FramePairSlicer has description."""
        from nodes.frame_pair_slicer import FramePairSlicer

        assert hasattr(FramePairSlicer, 'DESCRIPTION')
        assert isinstance(FramePairSlicer.DESCRIPTION, str)
        assert len(FramePairSlicer.DESCRIPTION) > 0

    def test_chunk_processor_has_docstring(self):
        """Test ChunkProcessor has docstring."""
        from nodes.chunk_processor import TLBVFI_ChunkProcessor

        assert TLBVFI_ChunkProcessor.__doc__ is not None
        assert len(TLBVFI_ChunkProcessor.__doc__) > 0
