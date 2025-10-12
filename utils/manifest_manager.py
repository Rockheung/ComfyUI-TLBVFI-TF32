"""
Manifest management for TLBVFI chunk-based processing.

Handles JSON-based chunk metadata tracking for resumable processing.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime


def create_session_id() -> str:
    """
    Create a unique session ID for chunk tracking.

    Returns:
        str: Session ID in format "tlbvfi_YYYYMMDD_HHMMSS"
    """
    return datetime.datetime.now().strftime("tlbvfi_%Y%m%d_%H%M%S")


def create_manifest(session_dir: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a new manifest file.

    Args:
        session_dir: Directory to store manifest
        metadata: Optional metadata dict

    Returns:
        dict: Created manifest
    """
    manifest = {
        'session_id': os.path.basename(session_dir),
        'created_at': datetime.datetime.now().isoformat(),
        'chunks': [],
        'metadata': metadata or {},
    }

    manifest_path = os.path.join(session_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_manifest(session_dir: str) -> Dict[str, Any]:
    """
    Load manifest from session directory.

    Args:
        session_dir: Directory containing manifest

    Returns:
        dict: Loaded manifest

    Raises:
        FileNotFoundError: If manifest doesn't exist
    """
    manifest_path = os.path.join(session_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any], session_dir: str):
    """
    Save manifest to session directory.

    Args:
        manifest: Manifest dict to save
        session_dir: Directory to save manifest
    """
    manifest_path = os.path.join(session_dir, "manifest.json")

    # Atomic write: write to temp file then rename
    temp_path = f"{manifest_path}.tmp"
    with open(temp_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    os.replace(temp_path, manifest_path)


def add_chunk_to_manifest(
    session_dir: str,
    chunk_id: int,
    chunk_path: str,
    shape: tuple,
    status: str = 'complete'
) -> Dict[str, Any]:
    """
    Add or update chunk entry in manifest.

    Args:
        session_dir: Session directory
        chunk_id: Chunk identifier
        chunk_path: Path to chunk file
        shape: Tuple of (num_frames, height, width, channels)
        status: Status string ('processing', 'complete', 'failed')

    Returns:
        dict: Updated manifest
    """
    # Load or create manifest
    manifest_path = os.path.join(session_dir, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = load_manifest(session_dir)
    else:
        manifest = create_manifest(session_dir)

    # Create chunk entry
    chunk_entry = {
        'chunk_id': chunk_id,
        'path': os.path.abspath(chunk_path),
        'shape': list(shape),
        'num_frames': shape[0],
        'status': status,
        'updated_at': datetime.datetime.now().isoformat(),
    }

    # Update or append
    existing_index = None
    for i, chunk in enumerate(manifest['chunks']):
        if chunk['chunk_id'] == chunk_id:
            existing_index = i
            break

    if existing_index is not None:
        manifest['chunks'][existing_index] = chunk_entry
    else:
        manifest['chunks'].append(chunk_entry)

    # Sort by chunk_id
    manifest['chunks'].sort(key=lambda x: x['chunk_id'])

    # Save
    save_manifest(manifest, session_dir)

    return manifest


def get_chunk_paths(session_dir: str) -> List[str]:
    """
    Get ordered list of chunk paths from manifest.

    Args:
        session_dir: Session directory

    Returns:
        list: Ordered chunk file paths
    """
    manifest = load_manifest(session_dir)
    chunks = sorted(manifest['chunks'], key=lambda x: x['chunk_id'])
    return [chunk['path'] for chunk in chunks if chunk.get('status') == 'complete']


def get_session_stats(session_dir: str) -> Dict[str, Any]:
    """
    Get statistics from session manifest.

    Args:
        session_dir: Session directory

    Returns:
        dict: Statistics including total_chunks, total_frames, etc.
    """
    manifest = load_manifest(session_dir)
    chunks = manifest['chunks']

    complete_chunks = [c for c in chunks if c.get('status') == 'complete']

    total_frames = sum(c['num_frames'] for c in complete_chunks)

    resolution = None
    if complete_chunks:
        shape = complete_chunks[0]['shape']
        resolution = f"{shape[2]}x{shape[1]}"  # width x height

    return {
        'session_id': manifest['session_id'],
        'created_at': manifest['created_at'],
        'total_chunks': len(chunks),
        'complete_chunks': len(complete_chunks),
        'total_frames': total_frames,
        'resolution': resolution,
        'metadata': manifest.get('metadata', {}),
    }


def cleanup_session(session_dir: str, delete_chunks: bool = True, delete_manifest: bool = False):
    """
    Clean up session files.

    Args:
        session_dir: Session directory
        delete_chunks: If True, delete chunk files
        delete_manifest: If True, delete manifest file
    """
    if not os.path.exists(session_dir):
        return

    if delete_chunks:
        try:
            manifest = load_manifest(session_dir)
            for chunk in manifest['chunks']:
                chunk_path = chunk['path']
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                    print(f"Deleted chunk: {chunk_path}")
        except Exception as e:
            print(f"Warning: Error deleting chunks: {e}")

    if delete_manifest:
        manifest_path = os.path.join(session_dir, "manifest.json")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
            print(f"Deleted manifest: {manifest_path}")

    # Try to remove directory if empty
    try:
        os.rmdir(session_dir)
        print(f"Removed session directory: {session_dir}")
    except OSError:
        pass  # Directory not empty or other error
