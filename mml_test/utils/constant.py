from string import Template

MAX_TOKENS = 32768
GENERATION_TEMPERATURE = 1.0
GENERATION_SEED = 215


# 识别推理 ok
OPEN_ENDED_COT_PROMPT1 = Template("""
Task: You are an experienced medical professional with expertise in analyzing various medical images. Given a medical image, a clinical context (patient age, symptoms, history), and a claim, determine whether the claim is reasonably observable and consistent in the image.

Claim: $claim
Clinical Context: $context

Instructions:
1. The image contains the target anatomical region for evaluation. Only judge based on visual findings presented in the current single image.
2. Refer to standard radiological visual comparison: evaluate signal intensity, density, anatomical location, left/right side, and morphological performance by comparing with adjacent normal tissue in the same slice.
3. Clinical context is for reference only. Do not infer, speculate, or add diagnoses that cannot be visually supported by the image.
4. Answer yes if the claim is generally consistent with visible imaging manifestations, including subtle or relative abnormal signals/densities that conform to routine radiological judgment.
5. Answer no only when the description is obviously contradictory to the image, the mentioned structure or lesion is completely absent, or the content requires other sequences/slices to verify.
6. For MRI signal judgment: terms such as hyperintensity / hypointensity adopt relative comparison standards with surrounding normal tissues on the same sequence, not limited to extremely obvious or striking abnormal changes.
7. If there is no image, it is also normal. Then please answer directly based on the text.

Think step by step inside <reasoning> tags. Then, after your reasoning, output ONLY the answer in the format:

Therefore, the final answer is: Answer: [Yes/No]

Do not add any other text after this line.
""")

# 并行推理
OPEN_ENDED_COT_PROMPT2 = Template("""
Task: You are an experienced medical professional with expertise in analyzing various medical images. Given some medical images, a clinical context (patient age, symptoms, history), and a claim, determine whether the claim is reasonably observable and consistent in the image.

Claim: $claim
Clinical Context: $context

Instructions:
1. The input may contain one or multiple images. You must integrate information from ALL specified sources (whether from one image or multiple images) to make your judgment.
2. For cross-modality claims (e.g., "CT and MRI both show..."): Verify that the finding is present in each specified modality. Different modalities may reveal different aspects of the same finding.
3. For multi-location claims (e.g., "both lungs show..." or "bilateral thalami show..."): Verify that the finding is present in each specified anatomical location. These locations may be visible within a single image or across multiple images.
4. For multi-sequence claims (e.g., lesion shows specific signal patterns across sequences): Verify that the described characteristics are consistent with the findings across the provided sequences.
5. Answer yes only if ALL specified modalities/locations demonstrate findings consistent with the claim, using standard radiological judgment criteria (signal intensity, density, morphology compared to adjacent normal tissue).
6. Answer no if ANY of the specified modalities/locations lack the claimed finding, or if the description is obviously contradictory to the findings in any of the provided images.
7. Do not infer, speculate, or add diagnoses that cannot be visually supported by the provided images. Clinical context is for reference only.
8. If there is no image, it is also normal. Then please answer directly based on the text.

Think step by step inside <reasoning> tags. Then, after your reasoning, output ONLY the answer in the format:

Therefore, the final answer is: Answer: [Yes/No]

Do not add any other text after this line.
""")

# 测量推理
OPEN_ENDED_COT_PROMPT3 = Template("""
Task: You are an experienced medical professional with expertise in analyzing various medical images. Given a medical image, a clinical context (patient age, symptoms, history), and a claim, determine whether the claim is reasonably observable and consistent in the image.

Claim: $claim

Instructions:
1. The image shows a fetal head ultrasound. Your task is to verify whether the claimed head circumference (HC) measurement is reasonably accurate based on the visible anatomical structures.
2. Assess the claim using standard fetal ultrasound measurement principles: identify the correct calvarial boundaries (outer-to-outer), recognize the standard elliptical tracing plane (transventricular plane with thalami and cavum septi pellucidi), and evaluate whether the claimed measurement falls within plausible anatomical limits.
3. However, note that you are a radiologist performing visual inspection, not performing exact pixel-level measurement. You should judge whether the claimed value is visibly and clinically plausible consistent with the visible calvarial size and shape in the image, not whether it matches exact pixel tracing.
4. Answer yes if the claimed measurement is reasonably consistent with the visible calvarial size (allowing for typical observer variation of approximately ±5.15 mm based on Wright et al., 2020), the anatomy appears appropriate for the measurement, and there is no visible anatomical distortion or improper calvarial boundary detection.
5. Answer no if the claimed value is obviously implausible for the visible calvarial size in the image (e.g., grossly too small or too large suggesting potential measurement error), the calvarial boundaries are not clearly visible for proper measurement, or there is significant discrepancy between the claimed value and visible anatomical scale that cannot be explained by typical measurement variability.
6. Clinical context (gestational age, prior measurements if provided) is for reference only. Do not infer additional clinical information not visible in the image.
7. If there is no image, it is also normal. Then please answer directly based on the text.

Think step by step inside <reasoning> tags. Then, after your reasoning, output ONLY the answer in the format:

Therefore, the final answer is: Answer: [Yes/No]

Do not add any other text after this line.

""")

# 测量推理（结构化提示）
OPEN_ENDED_COT_PROMPT3_1 = Template("""
Task: You are an experienced medical professional with expertise in analyzing various medical images. Given a medical image, a clinical context (patient age, symptoms, history), and a claim, determine whether the claim is reasonably observable and consistent in the image.

Claim: $claim

Instructions:
1. The image shows a fetal head ultrasound. Your task is to verify whether the claimed head circumference (HC) measurement is reasonably accurate based on the visible anatomical structures.
2. Assess the claim using standard fetal ultrasound measurement principles: identify the correct calvarial boundaries (outer-to-outer), recognize the standard elliptical tracing plane (transventricular plane with thalami and cavum septi pellucidi), and evaluate whether the claimed measurement falls within plausible anatomical limits.
3. However, note that you are a radiologist performing visual inspection, not performing exact pixel-level measurement. You should judge whether the claimed value is visibly and clinically plausible consistent with the visible calvarial size and shape in the image, not whether it matches exact pixel tracing.
4. Answer yes if the claimed measurement is reasonably consistent with the visible calvarial size (allowing for typical observer variation of approximately ±5.15 mm based on Wright et al., 2020), the anatomy appears appropriate for the measurement, and there is no visible anatomical distortion or improper calvarial boundary detection.
5. Answer no if the claimed value is obviously implausible for the visible calvarial size in the image (e.g., grossly too small or too large suggesting potential measurement error), the calvarial boundaries are not clearly visible for proper measurement, or there is significant discrepancy between the claimed value and visible anatomical scale that cannot be explained by typical measurement variability.
6. Clinical context (gestational age, prior measurements if provided) is for reference only. Do not infer additional clinical information not visible in the image.
7. If there is no image, it is also normal. Then please answer directly based on the text.

You must strictly follow the fixed four-step reasoning process below to complete your judgment:
Step 1: Observe the ultrasound image and identify the contour of the fetal head circumference.
Step 2: Extract the claimed head circumference value and pixel scale information from the claim statement.
Step 3: Compare the claimed measurement with the visual anatomical size of the fetal head in the image, and judge whether the value is clinically reasonable within the allowable error range of ±5.15 mm.
Step 4: Synthesize the above visual observation and numerical rationality analysis to give the final judgment result.



Think step by step inside <reasoning> tags. Then, after your reasoning, output ONLY the answer in the format:

Therefore, the final answer is: Answer: [Yes/No]

Do not add any other text after this line.

""")

# 医学推理
OPEN_ENDED_COT_PROMPT4 = Template("""
Task: You are an experienced medical professional with expertise in analyzing various medical images and integrating them with clinical information to reach diagnostic conclusions or treatment recommendations. Given a medical image, patient clinical information (history, symptoms, examination findings), a question, and multiple choice options, select the single best answer.

Question: $claim

Instructions:
1. The image contains the relevant anatomical region for evaluation. Integrate visual findings from the image with the provided clinical information.
2. Consider relevant medical knowledge, including differential diagnoses, disease presentations, and standard clinical reasoning pathways.
3. For image interpretation: evaluate signal intensity, density, anatomical location, left/right side, and morphological features by comparing with adjacent normal tissue or expected normal findings.
4. For multi-modality images: integrate information across all provided images to form a complete clinical picture.
5. Answer based on the combined evidence from the image(s) and clinical information. Do not rely solely on clinical pattern recognition without visual confirmation, nor solely on image findings without clinical context.
6. Select the single best answer from the options provided. The correct answer is the one most consistent with the integrated analysis of all available information.

Think step by step inside <reasoning> tags. First, analyze the key findings from the image(s). Second, identify relevant clinical information. Third, integrate both sources to narrow down the options. Finally, select the best answer. Then, output ONLY the answer in the format:

Therefore, the answer is: [ A, B, C, D, or E]

Do not add any other text after this line. 
""")


COT_PROMPT = {
    "recognition": OPEN_ENDED_COT_PROMPT1, #识别推理。
    "parallel":OPEN_ENDED_COT_PROMPT2, #并行推理。
    #"measure":OPEN_ENDED_COT_PROMPT3, #测量推理。
    "measure":OPEN_ENDED_COT_PROMPT3_1, #测量推理。（结构化提示）
    "medical":OPEN_ENDED_COT_PROMPT4 #医学推理。
}

MODEL_PATH= "C:/Users/jzs/icd/model/" ;