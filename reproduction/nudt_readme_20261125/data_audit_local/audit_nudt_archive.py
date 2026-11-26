#!/usr/bin/env python3
"""Read-only audit of the downloaded NUDT archive and its extracted dataset.

Writes evidence only beneath this script's directory. Does not extract data,
change inputs, install packages, train, or invoke a GPU.
"""
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DATA = Path('/home/ly/DeepPro-main/datasets/NUDT-MIRSDT')
ARCHIVE = Path('/home/ly/ReSight-IR/data/public_archives/NUDT-MIRSDT.zip')
RECEIPT = ARCHIVE.with_suffix('.download.json')
OFFICIAL = Path('/home/ly/DeepPro-main')
PRIOR = Path('/home/ly/ReSight-IR/results/deeppro_plus_official/INPUTS.json')
OUTPUT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(name, value):
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    errors = []
    receipt = json.loads(RECEIPT.read_text())
    readme = (OFFICIAL / 'README.md').read_text()
    source_url = receipt['source_url']
    source_id = re.search(r'/file/d/([^/]+)/', source_url).group(1)
    readme_source_lines = [line for line in readme.splitlines() if f'/file/d/{source_id}/' in line]
    if not readme_source_lines:
        errors.append('Receipt source ID not found in the official README')
    chunk_results = []
    archive_hash = hashlib.sha256()
    expected_offset = 0
    with ARCHIVE.open('rb') as stream:
        for start, item in sorted(receipt['chunks'].items(), key=lambda pair: int(pair[0])):
            start = int(start)
            if start != expected_offset:
                errors.append(f'Download chunks not contiguous at {start}')
            stream.seek(start)
            blob = stream.read(item['bytes'])
            digest = sha(blob)
            ok = len(blob) == item['bytes'] and digest == item['sha256']
            ok = ok and item['end'] == start + item['bytes'] - 1
            if not ok:
                errors.append(f'Download chunk mismatch at {start}')
            archive_hash.update(blob)
            expected_offset = start + item['bytes']
            chunk_results.append({'offset': start, 'bytes': len(blob), 'sha256': digest, 'matches_receipt': ok})
    if ARCHIVE.stat().st_size != expected_offset or expected_offset != receipt['archive_bytes']:
        errors.append('Archive size differs from contiguous receipt chunks')
    records = []
    archive_files = set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or name.parts[0] != 'NUDT-MIRSDT':
                errors.append(f'Unexpected archive member path: {info.filename}')
                continue
            relative = str(name.relative_to('NUDT-MIRSDT'))
            if relative in archive_files:
                errors.append(f'Duplicate archive member: {relative}')
            archive_files.add(relative)
            # ZipFile.read validates each member CRC before returning bytes.
            archive_bytes = archive.read(info)
            archive_sha = sha(archive_bytes)
            local = DATA / relative
            local_bytes = local.read_bytes() if local.is_file() else None
            local_sha = sha(local_bytes) if local_bytes is not None else None
            matches = local_sha == archive_sha
            if not matches:
                errors.append(f'Extracted member differs or is missing: {relative}')
            records.append({'member': info.filename, 'bytes': info.file_size,
                            'archive_crc32': f'{info.CRC:08x}', 'archive_sha256': archive_sha,
                            'local_sha256': local_sha, 'matches': matches})
    current_files = {str(p.relative_to(DATA)) for p in DATA.rglob('*') if p.is_file()}
    extras = sorted(current_files - archive_files)
    missing = sorted(archive_files - current_files)
    if extras or missing:
        errors.append('Extracted dataset file set differs from archive')
    splits = {}
    for split in ('train', 'test'):
        path = DATA / f'{split}.txt'
        blob = path.read_bytes()
        lines = blob.decode().splitlines()
        names = list(dict.fromkeys(line.split('/')[0] for line in lines))
        official_matches = blob == (OFFICIAL / 'datasets/NUDT-MIRSDT' / path.name).read_bytes()
        if not official_matches:
            errors.append(f'{split} split differs from official source')
        splits[split] = {'frames': len(lines), 'sequences': len(names), 'sequence_names': names,
                         'sha256': sha(blob), 'byte_identical_to_official_split': official_matches,
                         'listed_paths_missing': [line for line in lines if not (DATA / line).is_file()]}
        if splits[split]['listed_paths_missing']:
            errors.append(f'{split} has missing listed MAT files')
    overlap = sorted(set(splits['train']['sequence_names']) & set(splits['test']['sequence_names']))
    if overlap:
        errors.append('Train/test sequence overlap')
    test_sequences = set(splits['test']['sequence_names'])
    test_records = [record for record in records if len(PurePosixPath(record['member']).parts) == 4
                    and PurePosixPath(record['member']).parts[1] in test_sequences
                    and PurePosixPath(record['member']).parts[2] in ('images', 'masks', 'masks_centroid')]
    test_digest = hashlib.sha256()
    for record in sorted(test_records, key=lambda record: record['member']):
        test_digest.update(f"{record['local_sha256']}  {record['member']}\n".encode())
    prior = json.loads(PRIOR.read_text())
    prior_match = test_digest.hexdigest() == prior['input_digest_sha256']
    if len(test_records) != 6000 or not prior_match:
        errors.append('Test input hash collection differs from prior official inference')
    counts = Counter()
    sequence_counts = Counter()
    for member in archive_files:
        parts = PurePosixPath(member).parts
        if len(parts) == 3 and parts[0].startswith('Sequence'):
            counts[parts[1]] += 1
            if parts[1] == 'images':
                sequence_counts[parts[0]] += 1
    summary = {
        'status': 'PASS' if not errors else 'FAIL',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_root': str(DATA), 'official_code_root': str(OFFICIAL),
        'archive': str(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'archive_sha256': archive_hash.hexdigest(),
        'download_receipt': str(RECEIPT), 'download_receipt_sha256': sha(RECEIPT.read_bytes()),
        'source_url': source_url, 'source_file_id': source_id,
        'official_readme_source_lines': readme_source_lines,
        'download_chunks_checked': len(chunk_results),
        'download_chunks_all_match_receipt': all(x['matches_receipt'] for x in chunk_results),
        'archive_file_count': len(archive_files), 'current_file_count': len(current_files),
        'all_archive_members_crc_valid': True,
        'all_current_files_sha256_match_archive': all(record['matches'] for record in records),
        'extra_local_files': extras, 'missing_local_files': missing,
        'sequence_count': len(sequence_counts), 'frames_per_sequence': dict(sorted(sequence_counts.items())),
        'sequence_file_counts_by_category': dict(counts), 'splits': splits,
        'train_test_sequence_overlap': overlap,
        'unlisted_sequences': sorted(set(sequence_counts) - set(splits['train']['sequence_names']) - test_sequences),
        'test_png_count': len(test_records), 'test_input_digest_sha256': test_digest.hexdigest(),
        'test_input_digest_matches_prior_official_inference': prior_match,
        'prior_input_evidence': str(PRIOR), 'prior_input_evidence_sha256': sha(PRIOR.read_bytes()),
        'script_sha256': sha(Path(__file__).read_bytes()),
        'referenced_but_missing_provenance_files': [str(p) for p in (
            Path('/home/ly/ReSight-IR/provenance/PUBLIC_ARCHIVE_NUDT-MIRSDT.json'),
            Path('/home/ly/ReSight-IR/provenance/PUBLIC_MEMBERS_NUDT-MIRSDT.json')) if not p.exists()],
        'claim_boundary': 'All extracted bytes match the locally retained downloaded ZIP; its receipt URL matches the official README link. No upstream-published cryptographic digest is asserted. This does not establish author training seed or historical software environment.',
        'errors': errors,
    }
    dump('members.json', records)
    dump('download_chunks.json', chunk_results)
    dump('summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('status', 'archive_sha256', 'archive_file_count',
        'sequence_count', 'download_chunks_all_match_receipt', 'all_current_files_sha256_match_archive',
        'test_input_digest_matches_prior_official_inference', 'errors')}, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
