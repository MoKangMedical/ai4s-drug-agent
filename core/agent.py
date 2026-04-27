"""
AI4S Drug Agent - 核心智能体模块
负责协调分子生成、筛选、逆合成分析的完整流程
"""

import os
import sys
import time
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.molecule_generator import MoleculeGenerator
from core.docking import DockingEngine
from core.retrosynthesis import RetrosynthesisPlanner
from core.evaluator import MoleculeEvaluator


class DrugDiscoveryAgent:
    """靶向药物发现智能体"""
    
    def __init__(self, config: Dict = None):
        """
        初始化智能体
        
        Args:
            config: 配置字典，包含LLM API密钥等
        """
        self.config = config or {}
        self.start_time = None
        self.log_messages = []
        
        # 初始化各模块
        logger.info("初始化智能体模块...")
        self.generator = MoleculeGenerator(config)
        self.docking_engine = DockingEngine(config)
        self.retro_planner = RetrosynthesisPlanner(config)
        self.evaluator = MoleculeEvaluator(config)
        
        # 结果存储
        self.generated_molecules = []
        self.filtered_molecules = []
        self.scored_molecules = []
        self.final_results = []
        
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.log_messages.append(log_entry)
        
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "DEBUG":
            logger.debug(message)
    
    def run(self, target_pdb: str, output_csv: str, num_molecules: int = 50):
        """
        执行完整的药物发现流程
        
        Args:
            target_pdb: 靶点PDB文件路径
            output_csv: 输出CSV文件路径
            num_molecules: 目标生成分子数量
        """
        self.start_time = time.time()
        self.log(f"开始执行任务，靶点: {target_pdb}")
        self.log(f"目标: 生成{num_molecules}个候选分子并规划合成路线")
        
        try:
            # 阶段1: 分析靶点结构
            self.log("=" * 60)
            self.log("阶段1: 分析靶点蛋白质结构")
            binding_site = self._analyze_target(target_pdb)
            
            # 阶段2: 生成候选分子
            self.log("=" * 60)
            self.log("阶段2: 生成候选药物分子")
            molecules = self._generate_molecules(binding_site, num_molecules * 3)
            
            # 阶段3: 虚拟筛选 - 对接打分
            self.log("=" * 60)
            self.log("阶段3: 虚拟筛选 - 分子对接")
            docked_molecules = self._virtual_screening(molecules, target_pdb)
            
            # 阶段4: 性质筛选
            self.log("=" * 60)
            self.log("阶段4: 分子性质筛选")
            filtered_molecules = self._filter_molecules(docked_molecules)
            
            # 阶段5: 逆合成分析
            self.log("=" * 60)
            self.log("阶段5: 逆合成路线规划")
            results = self._plan_synthesis(filtered_molecules[:num_molecules])
            
            # 阶段6: 输出结果
            self.log("=" * 60)
            self.log("阶段6: 生成最终结果")
            self._save_results(results, output_csv)
            
            # 输出统计
            elapsed_time = time.time() - self.start_time
            self.log(f"任务完成! 总耗时: {elapsed_time:.2f}秒")
            self.log(f"生成分子数: {len(molecules)}")
            self.log(f"通过筛选: {len(filtered_molecules)}")
            self.log(f"最终输出: {len(results)}")
            
        except Exception as e:
            self.log(f"执行出错: {str(e)}", "ERROR")
            raise
    
    def _analyze_target(self, target_pdb: str) -> Dict:
        """分析靶点结构，识别结合位点"""
        self.log("解析PDB文件...")
        
        # 使用RDKit或BioPython分析蛋白质结构
        binding_site = self.docking_engine.analyze_binding_site(target_pdb)
        
        self.log(f"识别到结合位点: 中心={binding_site['center']}, 尺寸={binding_site['size']}")
        return binding_site
    
    def _generate_molecules(self, binding_site: Dict, num_molecules: int) -> List[str]:
        """生成候选分子"""
        self.log(f"开始生成{num_molecules}个候选分子...")
        
        # 使用多种策略生成分子
        molecules = []
        
        # 策略1: 基于模板的生成
        self.log("策略1: 基于药效团模板生成...")
        template_mols = self.generator.generate_from_templates(binding_site, num_molecules // 3)
        molecules.extend(template_mols)
        self.log(f"  生成{len(template_mols)}个分子")
        
        # 策略2: 基于片段的生成
        self.log("策略2: 基于分子片段组装...")
        fragment_mols = self.generator.generate_from_fragments(binding_site, num_molecules // 3)
        molecules.extend(fragment_mols)
        self.log(f"  生成{len(fragment_mols)}个分子")
        
        # 策略3: LLM辅助生成
        self.log("策略3: LLM辅助分子设计...")
        llm_mols = self.generator.generate_with_llm(binding_site, num_molecules // 3)
        molecules.extend(llm_mols)
        self.log(f"  生成{len(llm_mols)}个分子")
        
        # 去重
        unique_molecules = list(set(molecules))
        self.log(f"去重后: {len(unique_molecules)}个独特分子")
        
        return unique_molecules
    
    def _virtual_screening(self, molecules: List[str], target_pdb: str) -> List[Tuple[str, float]]:
        """虚拟筛选 - 分子对接"""
        self.log(f"对{len(molecules)}个分子进行对接打分...")
        
        docked_results = []
        
        for i, mol_smiles in enumerate(molecules):
            if i % 10 == 0:
                self.log(f"  进度: {i}/{len(molecules)}")
            
            try:
                # 执行分子对接
                score = self.docking_engine.dock(mol_smiles, target_pdb)
                if score is not None:
                    docked_results.append((mol_smiles, score))
            except Exception as e:
                self.log(f"  对接失败 {mol_smiles}: {str(e)}", "WARNING")
                continue
        
        # 按结合能排序（越低越好）
        docked_results.sort(key=lambda x: x[1])
        
        self.log(f"成功对接: {len(docked_results)}个分子")
        if docked_results:
            self.log(f"最佳结合能: {docked_results[0][1]:.2f} kcal/mol")
        
        return docked_results
    
    def _filter_molecules(self, docked_molecules: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """分子性质筛选"""
        self.log(f"对{len(docked_molecules)}个分子进行性质筛选...")
        
        filtered = []
        
        for mol_smiles, score in docked_molecules:
            # 评估分子性质
            properties = self.evaluator.evaluate(mol_smiles)
            
            if properties is None:
                continue
            
            # 筛选条件
            if self._passes_filters(properties):
                filtered.append((mol_smiles, score, properties))
                self.log(f"  通过: {mol_smiles} (结合能: {score:.2f})", "DEBUG")
            else:
                self.log(f"  过滤: {mol_smiles} - {properties.get('filter_reason', '性质不达标')}", "DEBUG")
        
        # 综合评分排序
        filtered.sort(key=lambda x: x[1])  # 按结合能排序
        
        self.log(f"通过筛选: {len(filtered)}个分子")
        return [(mol, score) for mol, score, _ in filtered]
    
    def _passes_filters(self, properties: Dict) -> bool:
        """检查分子是否通过筛选条件"""
        # Lipinski's Rule of Five
        if properties.get('molecular_weight', 0) > 500:
            return False
        if properties.get('logp', 0) > 5:
            return False
        if properties.get('hbd', 0) > 5:
            return False
        if properties.get('hba', 0) > 10:
            return False
        
        # 可合成性评分
        if properties.get('sa_score', 0) > 6:  # SA score越低越好
            return False
        
        # 结构合理性
        if not properties.get('valid', False):
            return False
        
        return True
    
    def _plan_synthesis(self, molecules: List[Tuple[str, float]]) -> List[Dict]:
        """规划合成路线"""
        self.log(f"为{len(molecules)}个分子规划合成路线...")
        
        results = []
        
        for i, (mol_smiles, score) in enumerate(molecules):
            self.log(f"  分子 {i+1}/{len(molecules)}: {mol_smiles}")
            
            try:
                # 执行逆合成分析
                route = self.retro_planner.plan(mol_smiles)
                
                if route:
                    results.append({
                        'mol_smiles': mol_smiles,
                        'route': route,
                        'docking_score': score
                    })
                    self.log(f"    合成路线: {route[:100]}...")
                else:
                    self.log(f"    无法规划合成路线", "WARNING")
                    
            except Exception as e:
                self.log(f"    逆合成分析失败: {str(e)}", "WARNING")
                continue
        
        self.log(f"成功规划: {len(results)}条合成路线")
        return results
    
    def _save_results(self, results: List[Dict], output_csv: str):
        """保存结果到CSV文件"""
        self.log(f"保存结果到: {output_csv}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
        
        # 写入CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['mol_smiles', 'route'])
            
            for result in results:
                writer.writerow([result['mol_smiles'], result['route']])
        
        self.log(f"已保存{len(results)}条结果")
        
        # 保存日志
        log_file = output_csv.replace('.csv', '.log')
        self._save_log(log_file)
    
    def _save_log(self, log_file: str):
        """保存日志文件"""
        self.log(f"保存日志到: {log_file}")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.log_messages))
        
        self.log(f"日志已保存，共{len(self.log_messages)}条记录")


def main():
    """测试函数"""
    agent = DrugDiscoveryAgent()
    
    # 测试用例
    test_pdb = "data/target.pdb"
    if os.path.exists(test_pdb):
        agent.run(test_pdb, "output/test_result.csv", num_molecules=10)
    else:
        print(f"测试PDB文件不存在: {test_pdb}")


if __name__ == "__main__":
    main()
