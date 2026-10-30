"""Create a QUICK_SAVE kernel version (no public run), then optionally submit it.

Kaggle's CLI/kernels_push always creates a SAVE_AND_RUN_ALL version, which burns a
GPU session.  The internal SaveKernel request also accepts
KernelExecutionType.QUICK_SAVE: it stores the code as a version without running
it.  A code-competition submission reruns that version on the hidden test set,
so this path can submit without a public GPU run.
"""
import argparse
import json
import pathlib
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiSaveKernelRequest,
    KernelExecutionType,
)


def as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() == 'true'


def quick_save(api, folder):
    folder = pathlib.Path(folder)
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    code_file = folder / meta['code_file']
    text = code_file.read_text()
    owner, slug, version = api.parse_kernel_string(meta['id'])
    if version is not None:
        raise ValueError('kernel id must not contain a version')

    request = ApiSaveKernelRequest()
    request.id = meta.get('id_no')
    request.slug = meta['id']
    request.new_title = meta.get('title')
    request.text = text
    request.language = meta['language']
    request.kernel_type = meta['kernel_type']
    request.is_private = as_bool(meta.get('is_private'), True)
    request.enable_gpu = as_bool(meta.get('enable_gpu'), False)
    request.enable_tpu = as_bool(meta.get('enable_tpu'), False)
    request.enable_internet = as_bool(meta.get('enable_internet'), True)
    request.dataset_data_sources = list(meta.get('dataset_sources', []))
    request.competition_data_sources = list(meta.get('competition_sources', []))
    request.kernel_data_sources = list(meta.get('kernel_sources', []))
    request.model_data_sources = list(meta.get('model_sources', []))
    request.category_ids = list(meta.get('keywords', []))
    request.docker_image_pinning_type = meta.get('docker_image_pinning_type')
    request.docker_image = meta.get('docker_image')
    request.machine_shape = meta.get('machine_shape')
    request.kernel_execution_type = KernelExecutionType.QUICK_SAVE

    with api.build_kaggle_client() as kaggle:
        response = kaggle.kernels.kernels_api_client.save_kernel(request)
    return response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--message', default='quick-save code submission')
    ap.add_argument('--competition', default='name-that-llm')
    ap.add_argument('--output-name', default='submission.parquet')
    args = ap.parse_args()

    api = KaggleApi()
    api.authenticate()
    response = quick_save(api, args.folder)
    version = response.version_number or response.versionNumber
    print('QUICK_SAVE_RESPONSE', response.to_dict() if hasattr(response, 'to_dict') else response)
    print('QUICK_SAVE_ID', response.kernel_id or response.kernelId, 'VERSION', version)
    for field in ('invalid_dataset_sources', 'invalid_kernel_sources',
                  'invalid_model_sources', 'invalid_competition_sources'):
        value = getattr(response, field, None)
        if value:
            print('QUICK_SAVE_INVALID', field, value)
            return 2
    if args.submit and version:
        # Give the save a moment to become visible to the submission service.
        time.sleep(10)
        result = api.competition_submit_code(
            args.output_name, args.message, args.competition,
            f"{response.kernel_id or response.kernelId}",
            int(version), quiet=False,
        )
        print('SUBMIT_RESPONSE', result.to_dict() if hasattr(result, 'to_dict') else result)
        print('SUBMIT_REF', getattr(result, 'ref', None))
        print('SUBMIT_MSG', getattr(result, 'message', None))
    return 0


if __name__ == '__main__':
    sys.exit(main())
