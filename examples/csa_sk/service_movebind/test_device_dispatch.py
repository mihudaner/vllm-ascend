# SPDX-License-Identifier: Apache-2.0
"""Execute production loader/executor/hooks on CPU; never initializes an NPU."""
import ast
import copy
import importlib.util
import json
import os
from typing import Any, Callable
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import test_csa_aclnn_backend as legacy_backend
import test_csa_aclnn_service as legacy_service

ROOT = Path('/data00/wangkai/aclnnWorkspace/vllm-ascend-xllm-movebind')
CSA = ROOT / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd'
PACKAGE = 'vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd_tnd'


def extract(path, names, namespace, cls=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    selected = [copy.deepcopy(n) for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names), names
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class TestDevicePackage(unittest.TestCase):
    setUp = legacy_backend.TestCSAExecutor.setUp
    write_metadata = legacy_backend.TestCSAExecutor.write_metadata

    def device_package(self):
        (self.root/'lib/bytenn_pto_ops_npu.so').write_bytes(b'device-lib')
        (self.root/'dsf_csa_pto_bnn.run').write_bytes(b'device-opp')
        self.metadata.update(dispatch_mode='device', dispatch_schema=1, dispatch_source_commit="89f541752cd90ce7bee19ecfa9b1819754c6ecff", kernel_keys=[0],
                             policy_count=94, supported_policy_ids=list(range(94)),
                             interface_sha256=self.backend._INTERFACE_SHA256,
                             torch_library_sha256=self.backend._sha256(self.root/'lib/bytenn_pto_ops_npu.so'),
                             run_sha256=self.backend._sha256(self.root/'dsf_csa_pto_bnn.run'))
        self.write_metadata()
        return self.backend._read_package(self.root)

    def test_full94_package_has_one_static_family(self):
        package = self.device_package()
        self.assertEqual(set(package.policies.values()), set(range(94)))
        self.assertEqual(package.library.name, 'bytenn_pto_ops_npu.so')
        self.metadata['dispatch_source_commit'] = 'future-fix-provenance'
        self.write_metadata()
        self.backend._read_package(self.root)

    def test_incomplete_or_mismatched_capabilities_fail_closed(self):
        self.device_package()
        for key, wrong in [('dispatch_mode','host'), ('dispatch_schema',2), ('kernel_keys',[0,1]),
                           ('policy_count',62), ('supported_policy_ids',list(range(62))),
                           ('interface_sha256','bad'), ('torch','bad')]:
            with self.subTest(key=key):
                old=self.metadata.pop(key)
                self.write_metadata()
                with self.assertRaises(RuntimeError):
                    self.backend._read_package(self.root)
                self.metadata[key]=wrong
                self.write_metadata()
                with self.assertRaises(RuntimeError):
                    self.backend._read_package(self.root)
                self.metadata[key]=old
                self.write_metadata()

    def test_device_loader_uses_existing_package_and_caches_once(self):
        self.device_package()
        package=Mock()
        with patch.object(self.backend.importlib.util,'find_spec',return_value=types.SimpleNamespace(origin=str(self.root/'__init__.py'))) as find, patch.object(self.backend.importlib,'import_module',return_value=package) as imported, patch.object(torch.ops,'load_library') as loaded:
            self.backend.validate_backend_installation()
            self.backend.validate_backend_installation()
        find.assert_called_once_with('bytenn_pto_ops')
        imported.assert_called_once_with('bytenn_pto_ops')
        loaded.assert_called_once_with(str(self.root/'lib/bytenn_pto_ops_npu.so'))
        package.ensure_vendor.assert_called_once_with()

    def test_executor_never_selects_host_policy_and_calls_family_zero(self):
        package=self.device_package()
        prepared=types.SimpleNamespace(tokens=2,buffers={'x':torch.empty((2,4),dtype=torch.bfloat16)},allocations={},policy=None,execution_backend='aclnn')
        with patch.object(self.backend,'_load_kernel_package',return_value=package):
            executor=self.backend.AclnnExecutor(prepared)
        with patch.object(torch.ops.bytenn_pto_ops,'dsf_csa_pto_bnn',create=True) as device:
            executor()
            executor.rebind({'x':torch.empty((2,4),dtype=torch.bfloat16)})
            executor()
        self.assertEqual(device.call_count,2)
        self.assertEqual(device.call_args.args[1],0)
        prepared.policy=Mock()
        with patch.object(self.backend,'_load_kernel_package',return_value=package), self.assertRaises(RuntimeError):
            self.backend.AclnnExecutor(prepared)
        prepared.policy.constexprs.assert_not_called()


class TestDeviceService(unittest.TestCase):
    def test_configuration_requires_spmd(self):
        resolve=legacy_service.TestCSABackendConfiguration().resolve
        self.assertEqual(resolve('spmd_tnd',' ACLNN '),'aclnn')
        for variant in ('mpmd_tnd','performance'):
            with self.assertRaises(ValueError):
                resolve(variant,'aclnn')

    def test_model_loads_device_package_without_pypto_runtime(self):
        fixture=legacy_service.TestCSAACLNNService()
        fixture.setUp()
        backend=Mock()
        modules,model=fixture.model_dependencies('spmd_tnd',backend,execution='aclnn')
        with patch.dict(sys.modules,modules):
            fixture.hooks['prepare_csa_model'](model)
        modules[PACKAGE+'.backend'].validate_backend_installation.assert_called_once_with()
        fixture.hooks['init_pto_runtime'].assert_not_called()

    def graph_fixture(self,backend='aclnn'):
        config=load_file('service_config_cpu',CSA/'service_config.py')
        ns={'torch':torch,'MAX_QUERY_TOKENS':config.MAX_QUERY_TOKENS,
            'native_query_bound':config.native_query_bound,'supported_descriptor':config.supported_descriptor,
            'compact_metadata_capacity':lambda t,b:t+b,'_native_token_views':lambda *args:[]}
        extract(CSA/'graph_contract.py',['tensor_descriptor'],ns)
        extract(CSA/'service.py',['graph_signature','_binding_policy'],ns,'CSAServiceRuntime')
        runtime=types.SimpleNamespace(execution_backend=backend,batch_capacity=2,layer_name='compressed',wrapper=object(),
            prefixes={key:key for key in ('swa','compressed','state','indexer','indexer_state')},
            _select_policy=Mock(side_effect=lambda **kw:kw['max_seq_len']))
        runtime._binding_policy=types.MethodType(ns['_binding_policy'],runtime)
        runtime.graph_signature=types.MethodType(ns['graph_signature'],runtime)
        metadata={}
        for name in runtime.prefixes:
            req=types.SimpleNamespace(seq_lens=torch.zeros(2,dtype=torch.int32),query_start_loc=torch.zeros(3,dtype=torch.int32),
                max_seq_lens=8192,max_seqlen_q=1,num_reqs_actual=2,ori_win_right=None,ori_win_left=None,
                block_table=torch.zeros((2,1024),dtype=torch.int32),start_pos=torch.zeros(2,dtype=torch.int32),block_size=128)
            metadata[name]=types.SimpleNamespace(decode=req,num_prefills=0,num_decodes=2,num_actual_tokens=2,hadamard=None)
        context=types.SimpleNamespace(attn_metadata=metadata)
        dsa=types.ModuleType('vllm_ascend.ops.dsa')
        cache=torch.empty(2)
        dsa._build_kv_cache=lambda *args:[cache]
        return runtime,context,dsa

    def test_same_descriptors_replay_across_q_kv_changes_only_device_backend(self):
        for backend in ('aclnn','python'):
            runtime,context,dsa=self.graph_fixture(backend)
            with patch.dict(sys.modules,{'vllm_ascend.ops.dsa':dsa}):
                before=runtime.graph_signature(context,8)
                self.assertIsNotNone(before)
                for item in context.attn_metadata.values():
                    item.decode.max_seq_lens=32768
                    item.decode.max_seqlen_q=6
                    item.decode.seq_lens.fill_(32768)
                after=runtime.graph_signature(context,8)
                if backend=='aclnn':
                    self.assertEqual(before,after)
                    runtime._select_policy.assert_not_called()
                else:
                    self.assertNotEqual(before,after)

    def test_graph_still_rejects_unsupported_range_or_metadata_mismatch(self):
        runtime,context,dsa=self.graph_fixture()
        with patch.dict(sys.modules,{'vllm_ascend.ops.dsa':dsa}):
            for bad in (8191,1048577):
                for item in context.attn_metadata.values(): item.decode.max_seq_lens=bad
                self.assertIsNone(runtime.graph_signature(context,8))
            for item in context.attn_metadata.values(): item.decode.max_seq_lens=8192
            context.attn_metadata['state'].decode.max_seq_lens=16384
            self.assertIsNone(runtime.graph_signature(context,8))

    def test_graph_pointer_and_shape_remain_part_of_contract(self):
        runtime,context,dsa=self.graph_fixture()
        with patch.dict(sys.modules,{'vllm_ascend.ops.dsa':dsa}):
            before=runtime.graph_signature(context,8)
            context.attn_metadata['state'].decode.seq_lens=torch.zeros(2,dtype=torch.int32)
            self.assertNotEqual(before,runtime.graph_signature(context,8))
            context.attn_metadata['state'].decode.query_start_loc=torch.zeros(4,dtype=torch.int32)
            self.assertIsNone(runtime.graph_signature(context,8))

    def test_binding_key_ignores_contents_preserves_descriptor(self):
        ns={}
        extract(CSA/'service.py',['_descriptor_key'],ns)
        key=ns['_descriptor_key']
        a={'q':torch.tensor([0,1,2],dtype=torch.int32),'l':torch.tensor([8192,8192],dtype=torch.int32)}
        before=key(a)
        a['q'].copy_(torch.tensor([0,6,8]))
        a['l'].fill_(32768)
        self.assertEqual(before,key(a))
        a['l']=torch.zeros(3,dtype=torch.int32)
        self.assertNotEqual(before,key(a))

    def test_draft_capture_reserves_future_query_tokens(self):
        class Draft: num_query_per_req=6
        ns={'AscendDSparkProposer':Draft}
        path=ROOT/'vllm_ascend/worker/model_runner_v1.py'
        tree=ast.parse(path.read_text())
        cls=next(n.name for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='_get_tnd_capture_sequence_limit' for f in n.body))
        extract(path,['_get_tnd_capture_sequence_limit'],ns,cls)
        call=ns['_get_tnd_capture_sequence_limit']
        self.assertEqual(call(types.SimpleNamespace(drafter=Draft(),max_model_len=131072)),131066)
        self.assertEqual(call(types.SimpleNamespace(drafter=None,max_model_len=131072)),131072)


class TestStaticCompile(unittest.TestCase):
    def test_readonly_preflight_and_package_cache_identity(self):
        mod=load_file('static_compile_cpu',CSA/'static_compile.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            include=root/'pto-isa/include'
            (include/'pto/common').mkdir(parents=True)
            (include/'pto/pto-inst.hpp').write_text('header')
            (include/'pto/common/type.hpp').write_text('enum TFillPadMode {};')
            for vendor in ('dsf_csa_pto_bnn',):
                descriptor=root/f'ascend/vendors/{vendor}/op_impl/ai_core/tbe/{vendor}_impl/dynamic/{vendor}.py'
                descriptor.parent.mkdir(parents=True)
                descriptor.write_text(repr('-I'+str(include)))
                with patch.object(mod,'_package_root',return_value=root):
                    self.assertEqual(mod.prepare_static_compile(),include)
            manifest=root/'package_manifest.json'
            manifest.write_text('{"binary":"one"}')
            with patch.object(mod,'_package_root',return_value=root):
                first=mod.package_compile_identity()
                manifest.write_text('{"binary":"two"}')
                self.assertNotEqual(first,mod.package_compile_identity())
                (include/'pto/common/type.hpp').write_text('old header')
                with self.assertRaisesRegex(RuntimeError,'incompatible'):mod.prepare_static_compile()



class TestCompilerIntegration(unittest.TestCase):
    def dependencies(self, backend):
        ascend=types.SimpleNamespace(enable_npugraph_ex=True,enable_static_kernel=True,enable_super_kernel=True)
        ns={'Any':Any,'Callable':Callable,'AscendCompilationConfig':object,'VllmConfig':object,
            'get_ascend_config':lambda:types.SimpleNamespace(ascend_compilation_config=ascend),
            'logger':Mock(),'os':os,'_compute_decode_cudagraph_batch_sizes':lambda cfg:[8,16]}
        source=ROOT/'vllm_ascend/compilation/compiler_interface.py'
        extract(source,['_configure_backend'],ns)
        extract(source,['compute_hash'],ns,'AscendCompiler')
        pkg=types.ModuleType('vllm_ascend')
        pkg.envs=types.SimpleNamespace(VLLM_ASCEND_PTO_CSA_BACKEND=backend)
        helper=types.ModuleType(PACKAGE+'.static_compile')
        helper.prepare_static_compile=Mock(return_value=Path('/valid/pto-isa/include'))
        helper.package_compile_identity=Mock(return_value='package-one')
        npu=types.ModuleType('torch_npu')
        npu.__version__='test'
        return ns,ascend,pkg,helper,{'vllm_ascend':pkg,helper.__name__:helper,'torch_npu':npu}

    def test_normal_compiler_preflights_aclnn_once(self):
        cfg=types.SimpleNamespace(parallel_config=types.SimpleNamespace(local_world_size=8,data_parallel_size_local=2))
        for backend in ('aclnn','python'):
            for npugraph in (True,False):
                with self.subTest(backend=backend,npugraph=npugraph):
                    ns,ascend,_,helper,modules=self.dependencies(backend)
                    config=Mock()
                    processor=Mock() if npugraph else None
                    with patch.dict(sys.modules,modules), patch.dict(os.environ,{},clear=True):
                        ns['_configure_backend'](config,ascend,cfg,processor)
                        self.assertEqual(os.environ['LOCAL_WORLD_SIZE'],'16')
                    if backend=='python':helper.prepare_static_compile.assert_not_called()
                    else:helper.prepare_static_compile.assert_called_once_with()
                    if npugraph:
                        options=processor.call_args.args[1]['options']
                        self.assertTrue(options['static_kernel_compile'])
                        self.assertTrue(options['super_kernel_optimize'])
                        self.assertEqual(options['super_kernel_optimize_options'],{'dcci_after_kernel_end':['.*']})
                    else:
                        self.assertIs(config.experimental_config.aclgraph._aclnn_static_shape_kernel,True)
                        self.assertIs(config.experimental_config.aclgraph._super_kernel_optimize,True)
                        self.assertEqual(config.experimental_config.aclgraph._super_kernel_optimize_options,{'dcci_after_kernel_end':['.*']})

    def test_device_binary_identity_isolates_persistent_fx_cache(self):
        ns,_,pkg,helper,modules=self.dependencies('python')
        owner=types.SimpleNamespace()
        with patch.dict(sys.modules,modules):
            python_hash=ns['compute_hash'](owner,None)
            helper.package_compile_identity.assert_not_called()
            pkg.envs.VLLM_ASCEND_PTO_CSA_BACKEND='aclnn'
            device_hash=ns['compute_hash'](owner,None)
            self.assertNotEqual(device_hash,python_hash)
            helper.package_compile_identity.return_value='package-two'
            self.assertNotEqual(device_hash,ns['compute_hash'](owner,None))

class TestDirectBindingBounds(unittest.TestCase):
    def test_device_descriptor_keeps_bounds_without_host_policy(self):
        config=load_file('service_config_bounds_cpu',CSA/'service_config.py')
        ns={name:getattr(config,name) for name in ('MAX_QUERY_TOKENS','MAX_SEQUENCE_LENGTH','MIN_SEQUENCE_LENGTH')}
        ns['validation_error']=lambda message,**kwargs:message
        extract(CSA/'binding.py',['_validate_device_descriptor'],ns)
        valid=dict(tokens=96,requests=16,max_query_tokens=6,max_seq_len=32768,compressed_table_columns=2048)
        check=ns['_validate_device_descriptor']
        check(**valid)
        for key,value in [('tokens',0),('requests',False),('max_query_tokens',7),('max_seq_len',8191),('max_seq_len',1048577),('compressed_table_columns',0)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):check(**{**valid,key:value})


if __name__=='__main__': unittest.main()
