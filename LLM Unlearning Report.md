# A800 双卡上可快速完成的 LLM Unlearning + 教师教学学生建模实验方案

## 研究目标与可检验假设

你描述的流程，本质是在做一个“先去域知识（finance unlearning）→ 再用少量交互教学让学生恢复域能力”的可行性验证：如果能在很短时间内把“会写代码的学生”变成“对金融很陌生的学生”，再让“老师”用少量对话把关键金融能力教回来，那么它会是一个很强的学生建模范式（teacher 只需少量诊断与纠错即可）。这与近两年 LLM unlearning 研究的核心张力一致：**既要有效遗忘（forget），又要保留通用/其他能力（retain / utility），而且要可扩展**。citeturn15search0turn15search4turn14search3

为了让实验在几天内产出“有说服力的结论”，建议把目标具体化为下面三条可检验假设（每条都对应一个可量化指标）：

- **H1：可控遗忘（Domain Forgetting）**  
  经过 unlearning 后，学生模型在金融任务上的分数显著下降（接近随机/低基线），但在纯编程任务上的能力基本不变或仅小幅下降。citeturn15search0turn14search3

- **H2：可教性（Teachability After Unlearning）**  
  在不让老师接触测试题答案的前提下，老师与学生进行“少量回合”（例如 6–12 轮）教学之后，学生在金融测试集上的分数显著回升。该回升幅度应明显大于“无老师”的回升幅度。citeturn14search3turn14search2

- **H3：教学收益确实来自“学到了金融”，而不是“老师替你做题”**  
  需要对照组控制：让老师用同样轮数教学“未 unlearn 的学生”，比较回升；以及确保老师只用训练集/公开概念讲解，不直接喂测试题答案。否则实验会被“老师在对话里泄露正确解”污染。citeturn14search3turn3search1

特别提醒：大量研究指出，LLM unlearning 在“像没学过一样”的严格意义上很难达成；很多基线方法即使在标准基准（如 TOFU）上也会暴露遗忘不彻底或效用崩塌的问题，因此你的 prior study 更现实的定位应是：**验证“是否能得到足够强的 domain suppression（金融能力明显下降）且 coding 基本不掉”，以及在该 suppression 下 teacher 的少量教学是否能恢复任务表现**。citeturn3search1turn14search1turn15search0

## 资源约束下的模型与框架选型

你希望“2 张 A800 + 开源代码/模型/数据集 + 几天内完成”。这里最稳妥的工程路线是：学生用 7B 级别模型做 LoRA/QLoRA 级别的快速训练；老师用更强一些的开源指令模型做推理（不训练或只做极轻量微调），并用高吞吐推理引擎跑批对话与评测。

### 学生模型（coding agent）建议

优先选 **Qwen2.5-Coder 7B Instruct**（或同系列 7B/14B）：其开源权重在模型卡中明确为 Apache 2.0 体系，工程生态在 Hugging Face/Transformers 侧相对成熟，可降低许可证与可复现风险。citeturn2search9turn2search1turn2search29  
该系列的版权/许可信息中出现 **entity["company","Alibaba Cloud","cloud provider"]** 署名，适合做“尽量开源/宽松许可”前提下的实验选型。citeturn2search13turn2search1

备选（如你更偏“纯代码能力”）：

- **StarCoder2-7B/15B**：权重许可为 BigCode OpenRAIL-M（可用但包含使用限制，不是 OSI 意义上的“开源许可”）。citeturn2search3turn15search3turn15search31  
  这里的 **entity["organization","BigCode","open code llm project"]** 官方页面也明确说明 OpenRAIL-M 并非 OSI 定义的 open source license。citeturn15search3

不推荐作为“最省事默认选项”的（但你若已有现成权重也能用）：

- Code Llama / Llama 2 系：其社区许可证被 **entity["organization","Open Source Initiative","open source standards body"]** 明确指出不符合 OSI 开源定义（主要是附带使用限制）。citeturn2search2turn2search26  
  若你把“开源”理解为“权重可下载可研究”，当然仍可用；但若你要把实验产出放到更严格的“开源可复现”语境，Apache 2.0 的 Qwen 路线更省沟通成本。citeturn2search26turn2search9

### 老师模型（teacher）建议

为了把“老师的教学能力”与“学生的基础能力”拉开差距，建议老师用更大或更擅长指令遵循的开源模型（只推理即可）：

- Qwen2.5 系列 Instruct（例如 14B 或更高），同样走 Apache 2.0 许可路径，工程一致性最好。citeturn2search17turn2search9turn2search13

如果你坚持“老师也要训练/构建”，几天内可行的版本是：**只做 LoRA 级别的“教学风格对齐”（Socratic / rubric-based tutoring）**，而不是从头训练老师模型；否则会把项目时间拉爆。

### 关键工程组件（全部开源）

- **高吞吐推理**：vLLM（高吞吐、KV 管理、持续批处理等）适合跑“老师-学生批量对话”和大规模评测。citeturn12search2turn12search14turn12search26  
- **参数高效训练**：Hugging Face PEFT/LoRA（少量可训练参数）+ QLoRA（4-bit 量化底座 + LoRA 训练）适合在有限 GPU 下快速做 unlearning 与小样本再学习。citeturn12search1turn12search0turn12search17  
- **通用评测框架**：lm-evaluation-harness 可做标准化评测任务管理；OpenUnlearning 框架也集成了多种 unlearning 指标与基准，适合复现/快速对齐社区做法。citeturn12search3turn14search2turn4view0  

## 数据集与任务设计

你需要同时满足三件事：  
1) “金融能力”要可测；2) “编程能力”要可测（retain）；3) 教学阶段要能严格避免测试集泄露。

### 金融任务数据集（finance / job x）

建议用“两层任务”组合：一层是可自动评分的标准金融 QA/推理；另一层是更接近“工作交付物”的 GDPval 类任务。

**可自动评分（最适合几天内出结果）**

- **FinQA**：金融报告上的数值推理 QA 数据集，论文描述含 8,281 个样例、带程序/推理标注，且 GitHub/HF 均有可用版本，非常适合用作“金融能力曲线”的主指标。citeturn0search13turn5search10turn5search6  
- **TAT-QA**：表格+文本混合上下文的金融 QA（16,552 问题，2,757 个混合上下文），适合测“读取财报表格并做计算”的能力。citeturn5search3turn5search7turn5search23  
- **FinTextQA**：长文金融问答（1,262 个高质量 QA，来源于金融教材与政府网站），适合测“更解释型的金融知识与写作输出”。citeturn13search1turn13search9turn13search5  
- **Financial PhraseBank / FiQA-SA**（可选）：更偏金融情绪/观点任务；若你想让“金融遗忘”覆盖语言层面的金融语义，也可以加入少量样本作为 forget 集的一部分。citeturn0search21turn0search17turn13search0  

**工作流/交付物（贴近你提到的 GDPval workflows）**

- **openai/gdpval（Hugging Face 数据集）**：包含 220 条 gold tasks，跨 44 个职业；每条任务给出 prompt、参考文件（xlsx/pdf/docx 等）、deliverable 参考文件与 rubric（rubric_pretty / rubric_json）。citeturn10view0turn6view0turn1view3  
  这对你的“coding agent 做 finance 工作”非常契合：很多任务要求生成 Excel/PDF/Word 等交付物（例如会计/审计相关任务在数据集中出现频率很高），并且 rubric 中存在大量可程序化检查的条目（例如 workbook 名称、sheet 名、是否包含特定表格结构等），这使得“几天内做出相对客观的评分”成为可能。citeturn6view0turn1view3  
  同时也要注意：一些工具链（例如 Inspect Evals 的 GDPval 实现）默认依赖提交到在线自动评分服务的流程；若你要“纯开源闭环”，可只使用数据集与 rubric，自建本地评分器或挑选 rubric 可自动验证的子集。citeturn1view2turn10view0  

### 编程保留任务（retain / coding）

为了证明 unlearning 没把“写代码”能力打坏，至少要跑一个标准代码基准：

- **HumanEval**：OpenAI 发布的 164 道 Python 函数题及其评测 harness，适合测 pass@1 / pass@k。citeturn5search0turn5search8  
- **MBPP**：约 1k 道基础 Python 题目（带测试用例），可作为 retain 训练小集合或额外评测集。citeturn5search5turn5search1  

其中 GDPval 的不少任务本身也需要读写 xlsx/pdf/docx 并输出交付物脚本；你可以把这些“工具型脚本能力”当作 coding agent 的额外保留指标（但这会让目标更复杂，建议作为扩展项）。

## Unlearning 实现方案

这里给出一个“几天内最可能跑通”的路线：先用最简单但常见的 forget/retain 目标做 unlearning（LoRA/QLoRA），若出现崩塌或遗忘不足，再切换到更稳的开源实现（NPO 或 ULD）。

### 最小可行 unlearning：Forget–Retain 联合目标（LoRA/QLoRA）

核心思想：对 **forget 集**做“反向优化”（让模型更不会答），对 **retain 集**做正常 SFT（让模型继续会写代码），并且用小步数+小学习率避免整体崩掉。这种范式在 unlearning 文献与实践中很典型（很多方法都可视为在 forget/retain 间做多目标权衡）。citeturn15search0turn14search20

工程上建议用 **LoRA**（或 **QLoRA**）来加速与控风险：

- LoRA 只训练少量低秩适配参数，能显著减少需要更新的参数量与显存压力。citeturn12search1turn12search13  
- QLoRA 通过冻结 4-bit 量化底座 + 训练 LoRA，在有限 GPU 下可快速迭代，论文明确强调其显存效率。citeturn12search0turn12search4  

**推荐的“几天内可跑通”的数据构造：**

- Forget set（金融遗忘集）：从 FinQA + TAT-QA 训练集各采样 1k–5k 条，做成“指令→答案”的 SFT 格式（可让模型直接给最终数值/选项，或要求给 Python 代码+最终答案，但评测时只比最终答案）。citeturn5search10turn5search3  
- Retain set（编程保留集）：从 MBPP（或其他你已有的开源 code-instruct 小集合）采样 1k–5k 条，维持 coding skill。citeturn5search5turn5search1  

**训练配方（经验性，但足够先做 prior study）：**

- 适配器：r=8/16，alpha=16/32，dropout 0.05；target modules 先从 attention 投影层开始（不同模型命名略有差异）。citeturn12search1turn12search13  
- 损失：`L = L_retain - λ * L_forget`（λ 从 0.1–1.0 扫一圈，小步数先看曲线），并设定最大步数（例如 200–1000 step）看金融分数是否快速下降。  
- 关键：每隔固定步数就跑一次小规模评测（FinQA/TAT-QA 各 100 条 + HumanEval/MBPP 各 20 条），做早停：一旦 coding 掉太多就停。  

这种“以最小工程代价拿到可以解释的曲线”的做法，非常适合你要的“几天内 prior study”。

### 若出现崩塌或遗忘不足：切换到 NPO 或 ULD

大量工作指出，直接对 forget 集做梯度上升很容易出现 **catastrophic collapse（效用崩塌）** 或输出异常；**Negative Preference Optimization (NPO)** 被提出就是为了解决这类问题，其论文与开源代码都可用。citeturn14search1turn3search2turn14search31

- NPO 路线的优势：相对更稳，社区实现多，且 open-unlearning 等框架也支持多种基线方法对齐。citeturn4view0turn14search2  

另一个工程上很“快”的先进路线是 **Unlearning from Logit Difference (ULD)**：它通过训练一个“反向目标的 assistant LLM”，再用 logit difference 构造 unlearned model；作者提供了 MIT 许可的实现仓库，并明确支持添加新数据集。citeturn14search0turn1view0turn14search4  

如果你担心自己写训练脚本踩坑，ULD 的 repo 对“加新数据集/加新 loss”的说明相对直接，是几天内更现实的“用现成代码跑通”选项之一。citeturn1view0  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["large language model unlearning forget retain diagram","LoRA fine-tuning diagram transformer","teacher student tutoring LLM diagram","GDPval benchmark example task spreadsheet deliverable"],"num_per_query":1}

### 评测框架建议：借用 OpenUnlearning 的“指标思想”，但先做你需要的最小集

OpenUnlearning 提供了统一框架与多种方法/指标的基准化实现，强调“评测本身也很难，需要标准化与多指标”。citeturn14search2turn4view0  
对你的 prior study，不需要完全复现其所有指标（否则会超出几天范围），但建议吸收两条原则：

1) **永远同时报 forget 与 retain**（金融分数下降 + 代码分数不掉）。citeturn14search3turn15search0  
2) **至少留一个“抗投机”度量**：例如 GDPval 子集用结构化 rubric 检查，避免纯文本 judge 的不稳定。citeturn6view0turn10view0  

## 教师-学生交互与学习机制设计

你设想的“老师与 unlearn 后学生交互几轮，然后看学生能否做 finance”非常关键的一点是：**你要决定‘学习’发生在哪里**——是只靠上下文（in-context learning），还是允许学生做参数更新（tiny fine-tune / adapter update）。两者都能在几天内做，但结论含义不同。

### 交互协议建议：把“教学”与“考试”分离

为了避免老师泄露测试，建议把流程固定为三段：

- **遗忘前基线测评**：学生 S0 在 finance_test 与 code_test 上打分。  
- **unlearning 后测评**：学生 S− 再测一次，确认 finance 显著下降。  
- **教学阶段**：老师只用 finance_train（或你专门抽出的 tutoring set）与学生交互。  
- **教学后测评**：学生在 finance_test 上重新测；并且再测 code_test 看有没有副作用。

其中 finance_train/test 可以来自 FinQA/TAT-QA 官方划分；GDPval 没有自然 split，你可以对任务 id 做固定划分（例如按 task_id hash 切分）来保证复现。citeturn5search10turn5search3turn10view0  

### 教学的两种可行实现

**方案 A：纯 in-context 教学（最省事，最符合“几轮对话”设定）**

老师给学生一个很短的“金融解题模板 + 常用概念表 + 单位/比例坑点清单”，再用 3–5 道 finance_train 小题做演示与纠错，最后让学生独立做 finance_test。  
这种方案本质是测试：unlearning 是否让学生“缺失金融先验”，以及 teacher 是否能用少量 token 把关键先验补回去。它基本不用训练流程，很快。citeturn14search3turn12search2  

**方案 B：交互日志 → 小步 LoRA 再学习（更接近“学生真的学会了”）**

把老师与学生的对话结构化为 (instruction, response) 对：  
- instruction：老师讲解后的练习题/步骤提示  
- response：学生最终的解题代码/答案（若错，老师提供纠错后让学生再答一次，取“纠正后版本”作为 target）

然后对学生做一个非常短的 QLoRA 微调（例如 200–1000 step），再去跑 finance_test。QLoRA 的核心价值就是让这种“很小数据、很短训练”的实验可行。citeturn12search0turn12search4  

这两种方案都建议做，因为它们能回答两个不同问题：  
- A 回答“老师能不能靠上下文把学生带回金融任务”；  
- B 回答“老师生成的少量教学数据能不能让学生快速再学习”。

### 老师-学生调度方式（开源实现）

你可以直接用 Python 写一个最小 orchestrator（vLLM 起两个 endpoint：teacher 与 student），也可以用现成多智能体框架加快搭建：

- **entity["company","微软","software company"] Agent Framework**（MIT 许可）与 AutoGen 系生态提供多 agent 编排思路；不过 AutoGen 项目本身已进入维护模式，建议把它当作“参考/模板”，核心逻辑仍尽量保持可控与轻量。citeturn2search24turn2search0turn2search8  

## 评估方案、判定标准与几日内执行时间表

### 指标与判定标准

为了在几天内得出“是否值得继续”的结论，你需要一个清晰的 Go/No-Go 判定门槛。建议最少报告以下四个数：

1) **金融遗忘幅度**：`FinanceDrop = score(S0, finance_test) - score(S−, finance_test)`  
   - 主评测建议 FinQA + TAT-QA 各一个（不同类型：纯数值推理 vs 表文混合）。citeturn0search13turn5search3  

2) **编程保留幅度**：`CodeDrop = score(S0, code_test) - score(S−, code_test)`  
   - 主评测建议 HumanEval（或至少抽样 30–50 题快速跑）；必要时加 MBPP。citeturn5search0turn5search5  

3) **教学后回升**：  
   - 纯 ICL：`TeachGain_ICL = score(S−+teach_ctx, finance_test) - score(S−, finance_test)`  
   - LoRA 再学习：`TeachGain_LoRA = score(S−+teach_ft, finance_test) - score(S−, finance_test)`  

4) **对照组**：`score(S0+teach, finance_test)`  
   - 用于判断“老师教学本身对一个本来就会金融的模型提升多少”，从而把“unlearning 后可教性”与“普通 prompting 提升”区分开。citeturn14search3turn14search2  

**GDPval 的用法建议（保证几天内能客观评分）：**  
从 openai/gdpval 中筛 10–20 个与 finance/会计/审计强相关、且 rubric 中含大量“可程序检查条目”的任务（比如要求输出特定 Excel 文件结构、sheet 名、表头、单位等）。你可以直接写一个本地 checker：打开学生输出文件，逐条验证 rubric 中能够硬判定的条件并计分；对“需要主观判断”的条目先跳过或只做人工 spot-check。这样你能得到一个更贴近工作流的指标，而且完全不依赖闭源 judge。citeturn6view0turn10view0turn1view3  

### 几天内可落地的执行节奏（不依赖夜间长训）

下面是一套按“最小闭环优先”的顺序排列的执行清单；你可以在任何一步卡住时直接降级（例如先不做 GDPval、先不做 NPO/ULD），保证几天内一定有结论。

**准备阶段（半天到一天）**  
完成环境、数据、脚手架三件事：  
- vLLM 起 teacher 与 student 的推理服务；确保能批量生成与并发。citeturn12search2turn12search26  
- 拉取并抽样 FinQA、TAT-QA、HumanEval、MBPP；写统一的评测 runner（同一套 prompt 模板、同一套答案归一化规则）。citeturn5search10turn5search3turn5search0turn5search5  
- 先只跑 50–100 个样本做 smoke test，确认输出解析、分数统计、日志落盘都稳定。

**基线测评阶段（半天）**  
- 跑 S0：finance_test（小样本）+ code_test（小样本），得到基线。  
- 同时记录 generating 配置（温度、top_p、max_tokens、是否启用工具等），保证之后所有对照一致。

**unlearning 阶段（一天内）**  
- 先用“Forget–Retain 联合目标 + QLoRA”跑短训练，边跑边做小评测曲线。citeturn12search0turn12search1  
- 目标是在较少 step 内看到 finance_test 明显下降、code_test 基本不变。  
- 若出现输出崩坏/代码能力大幅下降，优先降低 λ、降低学习率、减少 forget 比重；若依然不稳，再考虑换 NPO。citeturn14search1turn14search31  

**教学阶段（半天到一天）**  
- 固定教学轮数（例如总 8 轮）：老师先给总结性策略，再做少量示例纠错。  
- 先做纯 ICL 教学评测（最快），得到 TeachGain_ICL。  
- 如果你想验证“真的学会”，再把对话日志整理为小数据集，对学生做 200–1000 step 的 LoRA 再学习，得到 TeachGain_LoRA。citeturn12search0turn12search4  

**GDPval 工作流加分项（可选，半天到一天）**  
- 从 openai/gdpval 选 10–20 个会计/审计/财务管理类任务（occupation/sector 字段可筛），跑以下两种设置：  
  - S0 vs S−（看 unlearning 是否让“工作流金融任务”下降）  
  - S−+teach（看教学是否恢复）  
- 只对“易自动验证”的 rubric 条目计分；保留原始文件输出供人工复核。citeturn6view0turn10view0turn1view3  

### 复现与开源交付物清单（建议你在项目目录里强制执行）

为了让你的 prior study 真正“可复现、可扩展”，建议你把输出固定为以下内容：

- `configs/`：所有超参、采样策略、prompt 模板版本化（建议用 YAML + git）。  
- `data_splits/`：明确列出每个数据集抽样的 id 列表（避免“重跑抽到不同题”）。  
- `models/`：保存 LoRA adapter 权重（而非完整权重更轻便），并记录 base model 的精确版本号。citeturn12search1turn12search37  
- `logs/`：每次评测输出 JSONL（题目 id、模型输出、解析后的答案、是否正确、耗时、token 数）。  
- `results/report.md`：汇总四组对照（S0、S−、S−+teach、S0+teach）在 finance/code/GDPval 子集上的分数与曲线。

只要把上述四组对照跑出来，即使分数“没达到你理想中的完全遗忘/完全恢复”，你也能非常清楚地判断：  
- unlearning 是否真的把金融压下去了；  
- 代价是多大的 coding 损失；  
- teacher 的少量交互究竟能补回多少（ICL vs 小步再学习）；  
- GDPval 工作流任务是否与 FinQA/TAT-QA 的结论一致。citeturn14search3turn3search1turn15search0