"""Generates a realistic, cross-lingual mock validation dataset for R2AI competition."""

import json
from pathlib import Path


def create_mock_dataset(output_dir: str | Path = "data/mock"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Articles Corpus (VI, EN with PMIDs, ZH, and negative distractors)
    articles = [
        # --- Urology / Nephrology ---
        {
            "doc_id": "vi_urology_001",
            "title": "Hướng dẫn chẩn đoán và xử trí sỏi niệu quản gây tắc nghẽn đường tiết niệu",
            "text": (
                "Sỏi niệu quản là một trong những nguyên nhân hàng đầu gây nên cơn đau quặn thận cấp và tắc nghẽn đường dẫn niệu. "
                "Bệnh nhân thường có biểu hiện đau dữ dội vùng mạn sườn thắt lưng lan xuống bộ phận sinh dục ngoài, kèm tiểu buốt rắt hoặc tiểu máu. "
                "Khi sỏi niệu quản gây tắc nghẽn hoàn toàn đường dẫn niệu kèm theo sốt hoặc dấu hiệu nhiễm khuẩn huyết, người bệnh cần được can thiệp giải áp cấp cứu bằng đặt ống thông niệu quản (sonde Double J) hoặc mở thận ra da qua hướng dẫn siêu âm. "
                "Sau khi tình trạng nhiễm trùng và ứ nước được giải quyết ổn định, bệnh nhân sẽ được đánh giá lại kích thước sỏi để lựa chọn tán sỏi nội soi ngược dòng hoặc tán sỏi ngoài cơ thể."
            ),
            "lang": "vi",
            "source": "vi_moh",
        },
        {
            "doc_id": "31976211",
            "title": "Emergency management of obstructive ureteral calculi and sepsis",
            "text": (
                "Obstructive urolithiasis is a critical condition requiring rapid clinical differentiation between uncomplicated colic and infected obstruction. "
                "Nonsteroidal anti-inflammatory drugs are recommended for immediate analgesic relief in renal colic. "
                "In cases of acute obstructive ureteral calculi with concurrent urinary tract infection, urgent decompression via retrograde ureteral stenting (Double-J stent) or percutaneous nephrostomy is the gold standard of care. "
                "Definitive stone removal using ureteroscopy or shock wave lithotripsy should be delayed until the patient is completely afebrile and antibiotics have eradicated the bacteremia."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_urology_001",
            "title": "输尿管结石致急性尿路梗阻的临床诊疗指南",
            "text": (
                "输尿管结石引起的急性尿路梗阻常伴随突发性剧烈肾绞痛、恶心呕吐以及血尿。超声与泌尿系低剂量平扫CT是确诊结石位置及评估肾积水程度的首选手段。"
                "对于输尿管结石引起的急性尿路梗阻合并感染患者，急诊逆行留置输尿管双J管或经皮肾造瘘引流尿液是挽救肾功能、防止感染性休克的首要治疗手段。"
                "在急症解除且全身抗感染治疗控制良好后，可择期行经尿道输尿管镜钬激光碎石取石术（URL）。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Cardiology: STEMI ---
        {
            "doc_id": "vi_cardio_002",
            "title": "Phác đồ cấp cứu và can thiệp nhồi máu cơ tim cấp có ST chênh lên (STEMI)",
            "text": (
                "Nhồi máu cơ tim cấp có ST chênh lên là biến cố tim mạch tối cấp do tắc nghẽn hoàn toàn một hoặc nhiều nhánh động mạch vành cấp máu cho cơ tim. "
                "Bệnh nhân nhồi máu cơ tim cấp ST chênh lên cần được dùng ngay liều tải kháng kết tập tiểu cầu kép (Aspirin 150-300mg kèm Ticagrelor 180mg hoặc Clopidogrel 300-600mg), đồng thời chuyển khẩn cấp đến phòng can thiệp tim mạch để can thiệp động mạch vành qua da (PCI) thì đầu trong vòng 120 phút từ khi tiếp cận y tế. "
                "Trường hợp không thể chuyển đến trung tâm có PCI trong 120 phút, liệu pháp tiêu sợi huyết nên được chỉ định trong vòng 10 phút sau khi chẩn đoán."
            ),
            "lang": "vi",
            "source": "vi_cardio_soc",
        },
        {
            "doc_id": "34433215",
            "title": "Reperfusion strategies in ST-segment elevation myocardial infarction",
            "text": (
                "Rapid identification and initiation of reperfusion therapy are paramount for reducing mortality in acute STEMI. "
                "All patients without contraindications should receive immediate dual antiplatelet therapy and anticoagulation. "
                "Primary percutaneous coronary intervention (PCI) is the preferred reperfusion therapy for patients with STEMI within 12 hours of symptom onset, provided it can be performed within 120 minutes of STEMI diagnosis. "
                "Fibrinolytic therapy within 30 minutes of medical contact should be administered if timely primary PCI is unavailable."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_cardio_002",
            "title": "急性ST段抬高型心肌梗死再灌注治疗专家共识",
            "text": (
                "急性ST段抬高型心肌梗死（STEMI）的核心救治策略在于最短时间内开通闭塞冠状动脉，挽救濒死心肌。"
                "急性ST段抬高型心肌梗死（STEMI）发病12小时内的患者应优先实施直接经皮冠状动脉介入治疗（PPCI），要求首次医疗接触至导丝通过病变时间（FMC-to-W）控制在120分钟以内。"
                "若预计转运PCI时间延误超过120分钟且无溶栓禁忌证，应在30分钟内给予静脉溶栓治疗。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Cardiology: HFrEF Four Pillars ---
        {
            "doc_id": "vi_cardio_003",
            "title": "Khuyến cáo cập nhật về chẩn đoán và điều trị suy tim phân suất tống máu giảm",
            "text": (
                "Suy tim phân suất tống máu giảm (HFrEF) được xác định khi người bệnh có triệu chứng suy tim kèm EF <= 40%. "
                "Phác đồ điều trị nội khoa nền tảng (tứ trụ) cho suy tim phân suất tống máu giảm (HFrEF) bao gồm: thuốc ức chế thụ thể angiotensin-neprilysin (ARNI) hoặc ức chế men chuyển (ACEi), thuốc chẹn beta giao cảm (như bisoprolol, carvedilol), thuốc kháng aldosterone (MRA như spironolactone) và thuốc ức chế SGLT2 (dapagliflozin hoặc empagliflozin). "
                "Việc khởi trị và chỉnh liều nhanh chóng 4 nhóm thuốc này giúp giảm đáng kể tỷ lệ tử vong và nhập viện vì suy tim tiến triển."
            ),
            "lang": "vi",
            "source": "vi_guideline",
        },
        {
            "doc_id": "34424333",
            "title": "The four pillars of guideline-directed medical therapy for HFrEF",
            "text": (
                "Recent international clinical trials have redefined standard pharmacological treatment in heart failure. "
                "Guideline-directed medical therapy for heart failure with reduced ejection fraction comprises four foundational pillars: an ARNI/ACEi, an evidence-based beta-blocker, a mineralocorticoid receptor antagonist (MRA), and an SGLT2 inhibitor. "
                "Simultaneous or rapid sequential initiation of all four classes provides cumulative risk reduction in all-cause mortality and hospital readmission."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_cardio_003",
            "title": "射血分数降低的心力衰竭新四联药物治疗方案",
            "text": (
                "慢性心力衰竭伴射血分数降低（HFrEF）的病理生理机制复杂，需要多种靶点协同干预。"
                "针对射血分数降低心力衰竭（HFrEF）患者的基础药物治疗被称为‘新四联’方案，包含血管紧张素受体脑啡肽酶抑制剂（ARNI）、β受体阻滞剂、醛固酮受体拮抗剂（MRA）以及SGLT2抑制剂。"
                "这四类药物均被大型随机临床试验证实能够降低患者心血管死亡和因心衰恶化住院的复合终点事件。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Pulmonology: Community-Acquired Pneumonia (CAP) ---
        {
            "doc_id": "vi_pulmo_004",
            "title": "Hướng dẫn chẩn đoán và điều trị viêm phổi mắc phải ở cộng đồng của Bộ Y tế",
            "text": (
                "Viêm phổi mắc phải tại cộng đồng (CAP) là tình trạng nhiễm trùng cấp tính của nhu mô phổi xuất hiện ngoài bệnh viện. "
                "Thang điểm CURB-65 đánh giá 5 yếu tố: Rối loạn ý thức (Confusion), Urê máu > 7 mmol/L, Tần số thở >= 30 lần/phút, Huyết áp tâm thu < 90 mmHg hoặc tâm trương <= 60 mmHg, và Tuổi >= 65. Với điểm CURB-65 từ 2 trở lên, bệnh nhân nên nhập viện điều trị với kháng sinh phối hợp giữa beta-lactam và macrolide hoặc quinolone hô hấp. "
                "Đánh giá lâm sàng sau 48-72 giờ dùng kháng sinh là rất cần thiết để xem xét hạ bậc hoặc đổi phác đồ kháng sinh."
            ),
            "lang": "vi",
            "source": "vi_moh",
        },
        {
            "doc_id": "31573350",
            "title": "Diagnosis and treatment of community-acquired pneumonia in adults",
            "text": (
                "Management of community-acquired pneumonia begins with objective assessment of disease severity to determine outpatient versus inpatient disposition. "
                "The CURB-65 score is widely recommended to stratify severity in community-acquired pneumonia. For hospitalized non-ICU patients, standard empiric antimicrobial regimens consist of a beta-lactam combined with a macrolide or respiratory fluoroquinolone monotherapy. "
                "Macrolides provide additional atypical coverage and immunomodulatory benefits in moderate to severe infections."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_pulmo_004",
            "title": "成人社区获得性肺炎诊治指南与CURB-65评分应用",
            "text": (
                "社区获得性肺炎（CAP）的严重度评估直接决定患者是门诊口服抗生素治疗还是收住入院。"
                "CURB-65评分是评估社区获得性肺炎（CAP）严重程度的关键工具。评分>=2分患者建议住院治疗，初始经验性抗感染方案推荐β-内酰胺类联合大环内酯类或单用呼吸喹诺酮类药物。"
                "对于重症入住ICU患者，需覆盖耐甲氧西林金黄色葡萄球菌与假单胞菌感染可能。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Gastroenterology: Variceal Bleeding ---
        {
            "doc_id": "vi_gastro_005",
            "title": "Xử trí cấp cứu xuất huyết tiêu hóa trên do tăng áp lực tĩnh mạch cửa",
            "text": (
                "Vỡ giãn tĩnh mạch thực quản là nguyên nhân đe dọa tính mạng hàng đầu ở bệnh nhân xơ gan mất bù. "
                "Xử trí ban đầu xuất huyết do vỡ giãn tĩnh mạch thực quản gồm: hồi sức thể tích thận trọng duy trì huyết sắc tố 7-8 g/dL, dùng sớm thuốc vận mạch giảm áp lực tĩnh mạch cửa (terlipressin, octreotide hoặc somatostatin), kháng sinh dự phòng cephalosporin thế hệ 3 (ceftriaxone) và tiến hành nội soi thắt vòng cao su tĩnh mạch thực quản trong vòng 12 giờ. "
                "Chèn bóng chèn Sengstaken-Blakemore hoặc đặt stent tự bung chỉ áp dụng như biện pháp tạm thời khi nội soi thất bại."
            ),
            "lang": "vi",
            "source": "vi_guideline",
        },
        {
            "doc_id": "33785465",
            "title": "Management of acute variceal bleeding in cirrhosis",
            "text": (
                "Acute esophageal variceal hemorrhage requires prompt hemodynamic resuscitation and timely endoscopic intervention. "
                "Initial therapy for acute esophageal variceal hemorrhage includes cautious volume resuscitation targeting hemoglobin 7 to 8 g/dL, immediate administration of vasoactive drugs (terlipressin or octreotide), antibiotic prophylaxis with ceftriaxone, and urgent endoscopic band ligation within 12 hours. "
                "Transjugular intrahepatic portosystemic shunt (TIPS) should be considered early in patients at high risk of treatment failure."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_gastro_005",
            "title": "肝硬化门静脉高压食管胃静脉曲张破裂出血诊治指南",
            "text": (
                "食管胃静脉曲张破裂出血是肝硬化终末期的主要致死并发症之一，要求快速启动多学科抢救。"
                "食管胃静脉曲张破裂急性出血的急救原则包括：限制性液体复苏维持血红蛋白在70-80 g/L之间，尽早应用降门脉压血管活性药物（特利加压素或生长抑素类似物），常规静脉给予第三代头孢菌素预防感染，并在12小时内行内镜下套扎术（EBL）。"
                "早期评估Baveno标准有助于筛选急诊经颈静脉肝内门体分流术（TIPS）候选人群。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Endocrinology: Severe Hypoglycemia ---
        {
            "doc_id": "vi_endo_006",
            "title": "Chẩn đoán và xử trí cấp cứu hạ đường huyết ở người bệnh đái tháo đường",
            "text": (
                "Hạ đường huyết là biến chứng cấp tính thường gặp nhất ở bệnh nhân đái tháo đường điều trị bằng insulin hoặc sulfonylurea. "
                "Hạ đường huyết nặng được định nghĩa khi glucose máu < 3.0 mmol/L (54 mg/dL) hoặc bệnh nhân có rối loạn ý thức cần trợ giúp từ người khác. Với bệnh nhân hôn mê mất ý thức, tuyệt đối không cho ăn uống qua đường miệng; cấp cứu bằng tiêm tĩnh mạch bolus 20-50 ml dung dịch Glucose 20-30%, hoặc tiêm bắp 1 mg Glucagon, sau đó truyền duy trì Glucose 5-10%. "
                "Sau khi bệnh nhân tỉnh táo, cần tìm kiếm nguyên nhân gây hạ đường huyết như bỏ bữa, quá liều thuốc hoặc suy thận cấp kèm theo."
            ),
            "lang": "vi",
            "source": "vi_guideline",
        },
        {
            "doc_id": "32554789",
            "title": "Clinical guidance for severe hypoglycemia in diabetes mellitus",
            "text": (
                "Hypoglycemia is a critical barrier to glycemic optimization in insulin-treated patients with diabetes. "
                "Severe hypoglycemia requiring external assistance is treated emergently with intravenous dextrose (25 to 50 mL of 50% dextrose) or intramuscular glucagon (1 mg) if intravenous access is unavailable, followed by a continuous infusion of 10% dextrose. "
                "Oral administration of fast-acting carbohydrates is strictly reserved for conscious patients capable of safe swallowing."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_endo_006",
            "title": "糖尿病患者严重低血糖的急救处理规范",
            "text": (
                "在糖尿病强化胰岛素治疗过程中，低血糖事件显著增加心脑血管意外及全因死亡风险。"
                "对于意识障碍的严重低血糖糖尿病患者，禁止经口喂食。应立即建立静脉通路，静脉推注50%葡萄糖注射液40-60 ml，或肌肉注射胰高血糖素1 mg；待神志恢复后继续静脉滴注10%葡萄糖以维持血糖平稳。"
                "监测血糖直至完全稳定，并针对病因调整降糖方案。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Neurology: Acute Ischemic Stroke (rt-PA) ---
        {
            "doc_id": "vi_neuro_007",
            "title": "Hướng dẫn điều trị tái tưới máu trong đột quỵ nhồi máu não cấp tính",
            "text": (
                "Thời gian là não trong cấp cứu đột quỵ thiếu máu não cấp. Càng trì hoãn can thiệp tái tưới máu thì số lượng tế bào thần kinh tổn thương không hồi phục càng lớn. "
                "Thuốc tiêu sợi huyết Alteplase (rt-PA) đường tĩnh mạch với liều 0.9 mg/kg (tối đa 90 mg, bolus 10% trong 1 phút và truyền 90% còn lại trong 60 phút) được chỉ định trong cửa sổ vàng 4.5 giờ kể từ thời điểm khởi phát triệu chứng ở bệnh nhân đột quỵ thiếu máu não cấp sau khi loại trừ xuất huyết não trên phim CT scanner. "
                "Với trường hợp tắc mạch máu lớn, can thiệp lấy huyết khối cơ học bằng dụng cụ nên được tiến hành song song hoặc kế tiếp."
            ),
            "lang": "vi",
            "source": "vi_moh",
        },
        {
            "doc_id": "30482750",
            "title": "Intravenous thrombolysis in acute ischemic stroke: guidelines and evidence",
            "text": (
                "Intravenous recombinant tissue plasminogen activator remains the benchmark medical reperfusion therapy for acute ischemic stroke. "
                "Intravenous alteplase (0.9 mg/kg, maximum 90 mg) administered within 4.5 hours of symptom onset significantly improves functional outcomes in patients with acute ischemic stroke without evidence of intracranial hemorrhage on non-contrast CT. "
                "Blood pressure must be controlled below 185/110 mmHg prior to thrombolysis initiation and monitored continuously."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_neuro_007",
            "title": "急性缺血性脑卒中静脉溶栓治疗指南",
            "text": (
                "缺血性卒中超早期救治的关键在于迅速评估并实施血管再通，挽救缺血半暗带。"
                "对发病在4.5小时内的急性缺血性脑卒中患者，经头颅CT排除脑出血后，应尽早给予重组人组织型纤溶酶原激活剂（rt-PA，阿替普酶）静脉溶栓治疗，剂量为0.9 mg/kg（最大剂量90 mg）。"
                "大血管闭塞患者需同步评估桥接机械取栓的适应证。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Neurology: Status Epilepticus ---
        {
            "doc_id": "vi_neuro_008",
            "title": "Quy trình cấp cứu xử trí trạng thái động kinh ở người lớn",
            "text": (
                "Trạng thái động kinh là tình trạng cơn co giật kéo dài liên tục trên 5 phút hoặc xuất hiện từ 2 cơn trở lên mà giữa các cơn bệnh nhân không hồi phục ý thức. "
                "Bước 1 (giai đoạn sớm 0-10 phút): tiêm tĩnh mạch Benzodiazepine (Lorazepam 4mg hoặc Diazepam 10mg tiêm chậm). Bước 2 (10-30 phút nếu cơn chưa dứt): truyền tĩnh mạch một trong ba thuốc chống động kinh thế hệ hai (Levetiracetam 60 mg/kg tối đa 4500mg, hoặc Valproate natri 40 mg/kg tối đa 3000mg, hoặc Fosphenytoin 20 mg PE/kg). Bước 3 (> 30 phút): gây mê hồi sức bằng Propofol, Midazolam hoặc Thiopental. "
                "Cần đảm bảo kiểm soát đường thở và hỗ trợ thông khí trong suốt quá trình dùng thuốc an thần."
            ),
            "lang": "vi",
            "source": "vi_moh",
        },
        {
            "doc_id": "31754020",
            "title": "Management of convulsive status epilepticus in adults",
            "text": (
                "Status epilepticus is a medical emergency that demands systematic step-wise escalation of antiepileptic therapy. "
                "Initial therapy (phase 1) consists of intravenous benzodiazepines, preferably lorazepam 4 mg. If seizures persist beyond 10-15 minutes (phase 2), second-line intravenous antiepileptic drugs include levetiracetam (60 mg/kg), fosphenytoin (20 mg PE/kg), or valproate sodium (40 mg/kg). "
                "Refractory status epilepticus lasting over 30 minutes requires continuous general anesthesia with midazolam or propofol in an intensive care setting."
            ),
            "lang": "en",
            "source": "pubmed",
        },
        {
            "doc_id": "zh_neuro_008",
            "title": "惊厥性癫痫持续状态的多学科规范化诊疗流程",
            "text": (
                "癫痫持续状态是神经科常见危急重症，发作持续时间越长，神经元死亡与脑损伤风险越高。"
                "惊厥性癫痫持续状态初始治疗首选静脉注射地西泮（10 mg）或劳拉西泮（4 mg）。若发作持续超过10分钟，进入第二阶段治疗，推荐静脉给予左乙拉西坦（60 mg/kg）或丙戊酸钠（40 mg/kg）；超过30分钟则需在ICU进行咪达唑仑或丙泊酚全麻诱导。"
                "治疗全程需严密监测脑电图与生命体征。"
            ),
            "lang": "zh",
            "source": "zh_guideline",
        },
        # --- Distractors / Negatives (Irrelevant topics to test discriminative power) ---
        {
            "doc_id": "distractor_001",
            "title": "Chế độ dinh dưỡng giảm cân và ăn kiêng ketogenic an toàn",
            "text": (
                "Chế độ ăn ketogenic (Keto) là phương pháp dinh dưỡng cắt giảm tối đa carbohydrate và tăng cường chất béo tốt. "
                "Khi cơ thể cạn kiệt nguồn glycogen dự trữ, gan bắt đầu chuyển hóa axit béo thành thể ketone để cung cấp năng lượng thay thế glucose. "
                "Người thực hiện keto cần uống nhiều nước và bổ sung điện giải đầy đủ để tránh các triệu chứng mệt mỏi ban đầu."
            ),
            "lang": "vi",
            "source": "general_nutrition",
        },
        {
            "doc_id": "distractor_002",
            "title": "Chăm sóc da và phòng ngừa viêm da cơ địa dị ứng ở trẻ nhỏ",
            "text": (
                "Viêm da cơ địa là bệnh viêm da mạn tính thường gặp ở trẻ sơ sinh và trẻ nhỏ, gây ngứa ngáy khó chịu và tổn thương da. "
                "Nguyên tắc cốt lõi trong kiểm soát viêm da cơ địa là dưỡng ẩm thường xuyên bằng kem làm mềm da không chứa hương liệu. "
                "Tránh tắm nước quá nóng và hạn chế để trẻ tiếp xúc với các dị nguyên phổ biến như lông thú cưng, bụi phấn hoa."
            ),
            "lang": "vi",
            "source": "general_pediatrics",
        },
        {
            "doc_id": "distractor_003",
            "title": "Chẩn đoán và phục hồi chức năng sau phẫu thuật tái tạo dây chằng chéo trước",
            "text": (
                "Đứt dây chằng chéo trước (ACL) là chấn thương phổ biến trong thể thao đối kháng như bóng đá và bóng rổ. "
                "Sau phẫu thuật nội soi tái tạo dây chằng, bệnh nhân cần tuân thủ lộ trình vật lý trị liệu gồm tập co cơ tĩnh cơ tứ đầu đùi, tập biên độ gấp duỗi khớp gối và tăng cường thăng bằng. "
                "Thời gian trở lại thi đấu thể thao đỉnh cao thường kéo dài từ 9 đến 12 tháng."
            ),
            "lang": "vi",
            "source": "sports_medicine",
        },
        {
            "doc_id": "distractor_004",
            "title": "Hướng dẫn tiêm chủng vắc xin phòng bệnh cúm mùa cho phụ nữ mang thai",
            "text": (
                "Phụ nữ mang thai có nguy cơ cao gặp các biến chứng nặng khi nhiễm virus cúm mùa do những thay đổi về miễn dịch và tim mạch. "
                "Tổ chức Y tế Thế giới khuyến cáo tiêm vắc xin cúm bất hoạt ở bất kỳ giai đoạn nào của thai kỳ để tạo kháng thể bảo vệ cả mẹ và thai nhi. "
                "Vắc xin cúm mùa có độ an toàn cao và không làm tăng nguy cơ sảy thai hay dị tật bẩm sinh."
            ),
            "lang": "vi",
            "source": "vaccine_guidelines",
        },
        {
            "doc_id": "distractor_005",
            "title": "Phòng ngừa và kiểm soát cơn đau do thoái hóa khớp gối nguyên phát",
            "text": (
                "Thoái hóa khớp gối là bệnh lý thoái hóa sụn khớp tiến triển thường gặp ở người trên 50 tuổi. "
                "Các biện pháp không dùng thuốc bao gồm giảm cân, tập bơi lội hoặc đạp xe nhẹ nhàng giúp giảm tải lực tác động lên khớp gối. "
                "Khi có đau nhiều, có thể sử dụng paracetamol hoặc gel bôi chống viêm tại chỗ trước khi cân nhắc dùng thuốc uống NSAIDs."
            ),
            "lang": "vi",
            "source": "rheumatology",
        },
    ]

    # 2. Validation Queries in Vietnamese
    queries = [
        {
            "id": 1,
            "query": "Cần xử trí như thế nào khi bệnh nhân bị tắc nghẽn đường tiết niệu do sỏi niệu quản và chỉ định đặt sonde JJ?",
        },
        {
            "id": 2,
            "query": "Xử trí ban đầu và chiến lược tái tưới máu ở bệnh nhân nhồi máu cơ tim cấp có ST chênh lên (STEMI)?",
        },
        {
            "id": 3,
            "query": "Phác đồ điều trị nội khoa nền tảng (tứ trụ) cho bệnh nhân suy tim phân suất tống máu giảm (HFrEF) gồm những thuốc nào?",
        },
        {
            "id": 4,
            "query": "Tiêu chuẩn đánh giá mức độ nặng CURB-65 và phác đồ kháng sinh ban đầu cho viêm phổi mắc phải tại cộng đồng?",
        },
        {
            "id": 5,
            "query": "Xử trí cấp cứu xuất huyết tiêu hóa do vỡ giãn tĩnh mạch thực quản ở người bệnh xơ gan?",
        },
        {
            "id": 6,
            "query": "Dấu hiệu nhận biết và cách cấp cứu cơn hạ đường huyết nặng ở bệnh nhân đái tháo đường đang dùng insulin?",
        },
        {
            "id": 7,
            "query": "Cửa sổ thời gian vàng và tiêu chuẩn chỉ định dùng thuốc tiêu sợi huyết Alteplase (rt-PA) trong đột quỵ thiếu máu não cấp?",
        },
        {
            "id": 8,
            "query": "Phác đồ dùng thuốc từng bước để cắt cơn trong trạng thái động kinh co giật liên tục?",
        },
    ]

    # Map doc_id to doc text for exact substring verification
    doc_text_map = {d["doc_id"]: d["text"] for d in articles}

    # 3. Ground Truth Data
    ground_truth = [
        {
            "id": 1,
            "relevant_docs": ["vi_urology_001", "31976211", "zh_urology_001"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_urology_001",
                    "chunk_text": "Khi sỏi niệu quản gây tắc nghẽn hoàn toàn đường dẫn niệu kèm theo sốt hoặc dấu hiệu nhiễm khuẩn huyết, người bệnh cần được can thiệp giải áp cấp cứu bằng đặt ống thông niệu quản (sonde Double J) hoặc mở thận ra da qua hướng dẫn siêu âm.",
                },
                {
                    "doc_id": "31976211",
                    "chunk_text": "In cases of acute obstructive ureteral calculi with concurrent urinary tract infection, urgent decompression via retrograde ureteral stenting (Double-J stent) or percutaneous nephrostomy is the gold standard of care.",
                },
                {
                    "doc_id": "zh_urology_001",
                    "chunk_text": "对于输尿管结石引起的急性尿路梗阻合并感染患者，急诊逆行留置输尿管双J管或经皮肾造瘘引流尿液是挽救肾功能、防止感染性休克的首要治疗手段。",
                },
            ],
        },
        {
            "id": 2,
            "relevant_docs": ["vi_cardio_002", "34433215", "zh_cardio_002"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_cardio_002",
                    "chunk_text": "Bệnh nhân nhồi máu cơ tim cấp ST chênh lên cần được dùng ngay liều tải kháng kết tập tiểu cầu kép (Aspirin 150-300mg kèm Ticagrelor 180mg hoặc Clopidogrel 300-600mg), đồng thời chuyển khẩn cấp đến phòng can thiệp tim mạch để can thiệp động mạch vành qua da (PCI) thì đầu trong vòng 120 phút từ khi tiếp cận y tế.",
                },
                {
                    "doc_id": "34433215",
                    "chunk_text": "Primary percutaneous coronary intervention (PCI) is the preferred reperfusion therapy for patients with STEMI within 12 hours of symptom onset, provided it can be performed within 120 minutes of STEMI diagnosis.",
                },
                {
                    "doc_id": "zh_cardio_002",
                    "chunk_text": "急性ST段抬高型心肌梗死（STEMI）发病12小时内的患者应优先实施直接经皮冠状动脉介入治疗（PPCI），要求首次医疗接触至导丝通过病变时间（FMC-to-W）控制在120分钟以内。",
                },
            ],
        },
        {
            "id": 3,
            "relevant_docs": ["vi_cardio_003", "34424333", "zh_cardio_003"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_cardio_003",
                    "chunk_text": "Phác đồ điều trị nội khoa nền tảng (tứ trụ) cho suy tim phân suất tống máu giảm (HFrEF) bao gồm: thuốc ức chế thụ thể angiotensin-neprilysin (ARNI) hoặc ức chế men chuyển (ACEi), thuốc chẹn beta giao cảm (như bisoprolol, carvedilol), thuốc kháng aldosterone (MRA như spironolactone) và thuốc ức chế SGLT2 (dapagliflozin hoặc empagliflozin).",
                },
                {
                    "doc_id": "34424333",
                    "chunk_text": "Guideline-directed medical therapy for heart failure with reduced ejection fraction comprises four foundational pillars: an ARNI/ACEi, an evidence-based beta-blocker, a mineralocorticoid receptor antagonist (MRA), and an SGLT2 inhibitor.",
                },
                {
                    "doc_id": "zh_cardio_003",
                    "chunk_text": "针对射血分数降低心力衰竭（HFrEF）患者的基础药物治疗被称为‘新四联’方案，包含血管紧张素受体脑啡肽酶抑制剂（ARNI）、β受体阻滞剂、醛固酮受体拮抗剂（MRA）以及SGLT2抑制剂。",
                },
            ],
        },
        {
            "id": 4,
            "relevant_docs": ["vi_pulmo_004", "31573350", "zh_pulmo_004"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_pulmo_004",
                    "chunk_text": "Thang điểm CURB-65 đánh giá 5 yếu tố: Rối loạn ý thức (Confusion), Urê máu > 7 mmol/L, Tần số thở >= 30 lần/phút, Huyết áp tâm thu < 90 mmHg hoặc tâm trương <= 60 mmHg, và Tuổi >= 65. Với điểm CURB-65 từ 2 trở lên, bệnh nhân nên nhập viện điều trị với kháng sinh phối hợp giữa beta-lactam và macrolide hoặc quinolone hô hấp.",
                },
                {
                    "doc_id": "31573350",
                    "chunk_text": "The CURB-65 score is widely recommended to stratify severity in community-acquired pneumonia. For hospitalized non-ICU patients, standard empiric antimicrobial regimens consist of a beta-lactam combined with a macrolide or respiratory fluoroquinolone monotherapy.",
                },
                {
                    "doc_id": "zh_pulmo_004",
                    "chunk_text": "CURB-65评分是评估社区获得性肺炎（CAP）严重程度的关键工具。评分>=2分患者建议住院治疗，初始经验性抗感染方案推荐β-内酰胺类联合大环内酯类或单用呼吸喹诺酮类药物。",
                },
            ],
        },
        {
            "id": 5,
            "relevant_docs": ["vi_gastro_005", "33785465", "zh_gastro_005"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_gastro_005",
                    "chunk_text": "Xử trí ban đầu xuất huyết do vỡ giãn tĩnh mạch thực quản gồm: hồi sức thể tích thận trọng duy trì huyết sắc tố 7-8 g/dL, dùng sớm thuốc vận mạch giảm áp lực tĩnh mạch cửa (terlipressin, octreotide hoặc somatostatin), kháng sinh dự phòng cephalosporin thế hệ 3 (ceftriaxone) và tiến hành nội soi thắt vòng cao su tĩnh mạch thực quản trong vòng 12 giờ.",
                },
                {
                    "doc_id": "33785465",
                    "chunk_text": "Initial therapy for acute esophageal variceal hemorrhage includes cautious volume resuscitation targeting hemoglobin 7 to 8 g/dL, immediate administration of vasoactive drugs (terlipressin or octreotide), antibiotic prophylaxis with ceftriaxone, and urgent endoscopic band ligation within 12 hours.",
                },
                {
                    "doc_id": "zh_gastro_005",
                    "chunk_text": "食管胃静脉曲张破裂急性出血的急救原则包括：限制性液体复苏维持血红蛋白在70-80 g/L之间，尽早应用降门脉压血管活性药物（特利加压素或生长抑素类似物），常规静脉给予第三代头孢菌素预防感染，并在12小时内行内镜下套扎术（EBL）。",
                },
            ],
        },
        {
            "id": 6,
            "relevant_docs": ["vi_endo_006", "32554789", "zh_endo_006"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_endo_006",
                    "chunk_text": "Hạ đường huyết nặng được định nghĩa khi glucose máu < 3.0 mmol/L (54 mg/dL) hoặc bệnh nhân có rối loạn ý thức cần trợ giúp từ người khác. Với bệnh nhân hôn mê mất ý thức, tuyệt đối không cho ăn uống qua đường miệng; cấp cứu bằng tiêm tĩnh mạch bolus 20-50 ml dung dịch Glucose 20-30%, hoặc tiêm bắp 1 mg Glucagon, sau đó truyền duy trì Glucose 5-10%.",
                },
                {
                    "doc_id": "32554789",
                    "chunk_text": "Severe hypoglycemia requiring external assistance is treated emergently with intravenous dextrose (25 to 50 mL of 50% dextrose) or intramuscular glucagon (1 mg) if intravenous access is unavailable, followed by a continuous infusion of 10% dextrose.",
                },
                {
                    "doc_id": "zh_endo_006",
                    "chunk_text": "对于意识障碍的严重低血糖糖尿病患者，禁止经口喂食。应立即建立静脉通路，静脉推注50%葡萄糖注射液40-60 ml，或肌肉注射胰高血糖素1 mg；待神志恢复后继续静脉滴注10%葡萄糖以维持血糖平稳。",
                },
            ],
        },
        {
            "id": 7,
            "relevant_docs": ["vi_neuro_007", "30482750", "zh_neuro_007"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_neuro_007",
                    "chunk_text": "Thuốc tiêu sợi huyết Alteplase (rt-PA) đường tĩnh mạch với liều 0.9 mg/kg (tối đa 90 mg, bolus 10% trong 1 phút và truyền 90% còn lại trong 60 phút) được chỉ định trong cửa sổ vàng 4.5 giờ kể từ thời điểm khởi phát triệu chứng ở bệnh nhân đột quỵ thiếu máu não cấp sau khi loại trừ xuất huyết não trên phim CT scanner.",
                },
                {
                    "doc_id": "30482750",
                    "chunk_text": "Intravenous alteplase (0.9 mg/kg, maximum 90 mg) administered within 4.5 hours of symptom onset significantly improves functional outcomes in patients with acute ischemic stroke without evidence of intracranial hemorrhage on non-contrast CT.",
                },
                {
                    "doc_id": "zh_neuro_007",
                    "chunk_text": "对发病在4.5小时内的急性缺血性脑卒中患者，经头颅CT排除脑出血后，应尽早给予重组人组织型纤溶酶原激活剂（rt-PA，阿替普酶）静脉溶栓治疗，剂量为0.9 mg/kg（最大剂量90 mg）。",
                },
            ],
        },
        {
            "id": 8,
            "relevant_docs": ["vi_neuro_008", "31754020", "zh_neuro_008"],
            "relevant_chunks": [
                {
                    "doc_id": "vi_neuro_008",
                    "chunk_text": "Bước 1 (giai đoạn sớm 0-10 phút): tiêm tĩnh mạch Benzodiazepine (Lorazepam 4mg hoặc Diazepam 10mg tiêm chậm). Bước 2 (10-30 phút nếu cơn chưa dứt): truyền tĩnh mạch một trong ba thuốc chống động kinh thế hệ hai (Levetiracetam 60 mg/kg tối đa 4500mg, hoặc Valproate natri 40 mg/kg tối đa 3000mg, hoặc Fosphenytoin 20 mg PE/kg). Bước 3 (> 30 phút): gây mê hồi sức bằng Propofol, Midazolam hoặc Thiopental.",
                },
                {
                    "doc_id": "31754020",
                    "chunk_text": "Initial therapy (phase 1) consists of intravenous benzodiazepines, preferably lorazepam 4 mg. If seizures persist beyond 10-15 minutes (phase 2), second-line intravenous antiepileptic drugs include levetiracetam (60 mg/kg), fosphenytoin (20 mg PE/kg), or valproate sodium (40 mg/kg).",
                },
                {
                    "doc_id": "zh_neuro_008",
                    "chunk_text": "惊厥性癫痫持续状态初始治疗首选静脉注射地西泮（10 mg）或劳拉西泮（4 mg）。若发作持续超过10分钟，进入第二阶段治疗，推荐静脉给予左乙拉西坦（60 mg/kg）或丙戊酸钠（40 mg/kg）；超过30分钟则需在ICU进行咪达唑仑或丙泊酚全麻诱导。",
                },
            ],
        },
    ]

    # Rigorous Verification: Check that every chunk_text is strictly an exact substring of the parent document
    print("Verifying exact substring compliance for all ground truth chunks...")
    for item in ground_truth:
        qid = item["id"]
        for c in item["relevant_chunks"]:
            did = c["doc_id"]
            ctext = c["chunk_text"]
            assert did in doc_text_map, f"[Query {qid}] doc_id {did} not in articles corpus!"
            assert ctext in doc_text_map[did], (
                f"[Query {qid}] chunk_text is NOT a substring of {did}!"
            )
            assert did in item["relevant_docs"], (
                f"[Query {qid}] doc_id {did} missing from relevant_docs!"
            )

    print("All chunks passed 100% exact substring verification!")

    # Write files
    articles_file = out_path / "articles_all.jsonl"
    with open(articles_file, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    print(f"Saved {len(articles)} articles to {articles_file}")

    queries_file = out_path / "queries_val.jsonl"
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Saved {len(queries)} queries to {queries_file}")

    gt_file = out_path / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"Saved ground truth for {len(ground_truth)} queries to {gt_file}")

    # Also save separate language subsets for crawler simulation tests
    vi_articles = [a for a in articles if a["lang"] == "vi"]
    zh_articles = [a for a in articles if a["lang"] == "zh"]
    en_articles = [a for a in articles if a["lang"] == "en"]

    with open(out_path / "articles_vi.jsonl", "w", encoding="utf-8") as f:
        for a in vi_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    with open(out_path / "articles_zh.jsonl", "w", encoding="utf-8") as f:
        for a in zh_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    with open(out_path / "articles_en.jsonl", "w", encoding="utf-8") as f:
        for a in en_articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(
        f"Generated language subsets: {len(vi_articles)} VI, {len(zh_articles)} ZH, {len(en_articles)} EN."
    )


if __name__ == "__main__":
    create_mock_dataset()
