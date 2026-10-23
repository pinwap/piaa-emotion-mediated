# Research and Paper Engineering Playbook

## Lessons from the emotion-mediated PIAA project

เอกสารนี้สกัด “วิธีคิดในการทำวิจัย” จากโปรเจกต์ emotion-mediated personalized image aesthetic assessment (PIAA) ตั้งแต่การตั้งคำถาม การออกแบบข้อมูลและโมเดล การเลือก loss และ hyperparameter การทำสถิติ การตรวจ faithfulness การเขียนโค้ดให้ตรวจสอบย้อนหลังได้ ไปจนถึงการเขียนเปเปอร์และออกแบบตารางกับกราฟ

นี่ไม่ใช่เพียงสรุปว่าโปรเจกต์ทำอะไร แต่เป็นคู่มือสำหรับตอบคำถามว่า:

- ก่อนเขียนโค้ด ควรล็อกคำถามทางวิทยาศาสตร์อย่างไร
- จะรู้ได้อย่างไรว่า baseline และ control เปรียบเทียบกันอย่างยุติธรรม
- จะป้องกัน leakage, pseudo-replication และ multiple comparisons อย่างไร
- จะไล่จากตัวเลขใน PDF กลับไปหา unit-level result และโค้ดได้อย่างไร
- จะเขียน claim ให้แข็งแรงโดยไม่เกินสิ่งที่การทดลองพิสูจน์ได้อย่างไร
- จะเปลี่ยนข้อผิดพลาดและผลลบให้กลายเป็นความรู้ที่ทำให้งานดีขึ้นได้อย่างไร

เนื้อหาส่วนที่ระบุว่า “ในโปรเจกต์นี้” เป็นข้อเท็จจริงเฉพาะของ PIAA ชุดนี้ ส่วนหลักคิดสามารถนำไปปรับใช้กับงาน machine learning และ human-centered AI อื่นได้

---

## 1. แก่นของการทำวิจัยที่ควรจำ

งานวิจัยที่น่าเชื่อถือไม่ใช่งานที่มีโมเดลซับซ้อนที่สุด แต่เป็นงานที่ทุกการตัดสินใจเชื่อมกันอย่างตรวจสอบได้:

> คำถาม → estimand → ข้อมูล → split → โมเดล → control → หน่วยวิเคราะห์ → สถิติ → รูป/ตาราง → claim

ถ้าข้อต่อใดข้อต่อหนึ่งไม่ตรงกัน งานจะดูแข็งแรงกว่าหรืออ่อนกว่าความจริงทันที ตัวอย่างเช่น:

- คำถามถามเรื่อง “สิ่งที่โมเดลใช้คำนวณจริง” แต่ ablation ใช้น้ำหนักที่ไม่ได้อยู่ใน execution path ทั้งหมด
- ตารางรายงาน 387 แถว แต่ความเป็นอิสระจริงมีเพียง 129 คน
- โมเดลทดสอบมี population prior แต่ baseline ไม่มี prior แบบเทียบเท่า
- เลือก hyperparameter จาก test users แล้วค่อยรายงานผลว่าเป็น held-out
- ผล non-significant ถูกเขียนว่า “เท่ากัน”
- ตารางใช้ Holm-corrected p-value แต่ข้อความยังอ้าง p-value ก่อน correction

หลักปฏิบัติที่สำคัญที่สุดมี 12 ข้อ:

1. **เขียนคำถามทางวิทยาศาสตร์ก่อนเขียน experiment**
2. **กำหนด estimand ให้ชัดว่าเราต้องการวัดค่าอะไร**
3. **เขียนสมการของโมเดลที่รันจริง ไม่ใช่สมการที่อยากให้มันหมายถึง**
4. **ตั้งชื่อทุก tensor พร้อม shape และแหล่งที่มา**
5. **แยก shared, population และ personal components ให้ชัด**
6. **ออกแบบ control ให้เปลี่ยนทีละหนึ่งปัจจัย**
7. **กำหนดหน่วยวิเคราะห์และโครงสร้างการพึ่งพาก่อนเลือกสถิติ**
8. **เลือก hyperparameter บน validation ที่จำลอง test protocol โดยไม่แตะ test**
9. **เก็บผลระดับ unit ก่อนสรุปเป็น mean, SEM, p-value**
10. **สร้างตารางและกราฟจากแหล่งข้อมูลเดียว ไม่คัดลอกเลขด้วยมือ**
11. **เขียน claim ตามขอบเขตของหลักฐาน ไม่ตามความรู้สึกที่อยากเล่า**
12. **freeze เวอร์ชันของข้อมูล โค้ด ผล และเอกสารพร้อม provenance**

---

## 2. One-page mental model ของโปรเจกต์นี้

### 2.1 คำถามหลัก

PIAA ต้องทำนายว่าคนคนหนึ่งจะชอบภาพหนึ่งมากเพียงใด ปัญหาคือ personalization ที่แม่นมักเป็น black box ผู้ใช้ไม่รู้ว่าโมเดลมองรสนิยมของเขาอย่างไร

แนวคิดของงานนี้คือใช้ **aesthetic emotions** เป็นภาษากลาง:

- ภาพกระตุ้นอารมณ์อะไร
- คนแต่ละคนให้น้ำหนักอารมณ์เหล่านั้นต่อความชอบอย่างไร

โมเดลจึงทำหน้าที่เป็นเครื่องมือวิทยาศาสตร์เพื่อถามว่า:

- ความแม่นยำที่เสียไปจาก semantic bottleneck มีเท่าไร
- สิ่งที่เสียเป็นผลจากการบีบมิติ หรือจากการบังคับให้มิติมีความหมาย
- สูตรส่วนบุคคลที่อ่านได้สะท้อนเจ้าของสูตร ภาพ และข้อมูลของเจ้าของจริงหรือไม่

### 2.2 โครงสร้างที่รันจริง

ให้:

- x เป็น visual feature ของภาพ
- h เป็น Stage 1 mediator
- z = h(x) เป็น predicted emotion vector 7 มิติหลัง standardization
- y_pop(x) เป็น population GIAA prediction จาก raw visual features
- delta_u เป็นน้ำหนัก personal residual 7 ค่า
- b_u เป็น per-user intercept ที่ไม่ถูก penalize

Anchor C ซึ่งเป็นโมเดลหลักคำนวณ:

    y_hat_u(x) = y_pop(x) + b_u + z(x)^T delta_u

ประเด็นที่ห้ามสับสน:

- y_pop(x) มาจาก raw visual feature branch
- delta_u คือน้ำหนักอารมณ์ของ personal residual branch ที่อยู่ใน execution path จริง
- โมเดลเต็มไม่ได้บังคับให้ population branch ผ่าน 7 emotions
- ดังนั้นห้ามเขียนว่า “ทุกส่วนของ prediction ผ่าน concept bottleneck”
- เขียนได้ว่า “personal correction is routed through seven named emotions”

ใน emotion space ยังมี:

    w_total,u = w_pop + delta_u

โดย w_pop เป็น Stage-2 emotion coefficients ที่ fit บน pooled training group ภายใต้ scaler ของ training group แต่สำหรับ Anchor C:

- w_total เป็นการอ่าน absolute personal mapping ใน emotion space
- w_total ไม่ใช่การ decomposition ของ raw-feature y_pop(x)
- การใช้ w_total ทำ concept attribution ตอบคำถามระดับ “สูตรใน emotion space”
- การใช้ delta_u ตอบคำถามระดับ “personal residual computation ที่รันจริง”

กฎทั่วไปคือ **ก่อนบอกว่าน้ำหนักอธิบาย prediction ต้องระบุว่ามันอยู่ใน computational path ไหน**

### 2.3 ผลลัพธ์หนึ่งหน่วยคืออะไร

หนึ่ง evaluation unit คือ:

    participant × domain

มีผู้เข้าร่วม 129 คนและ 3 domains จึงมี:

    129 × 3 = 387 user-domain units

แต่ 387 ไม่ได้แปลว่ามี 387 คน:

- domain-specific analysis มี 129 คนอิสระ
- pooled Average มี 387 observations ที่จัดกลุ่มเป็น 129 participant clusters
- การทำ SEM และ test ของ Average ต้องรักษาสาม domains ของคนเดียวกันไว้เป็น block

---

## 3. ภาษาสัญลักษณ์และชื่อที่ควรใช้ให้คงที่

ความสับสนจำนวนมากเกิดจากเปลี่ยนชื่อสิ่งเดียวกันไปมา จึงควรมี symbol table ก่อนเริ่มโค้ดหรือเขียนเปเปอร์

| Symbol/name | Shape | ความหมาย | Fit จากไหน |
|---|---:|---|---|
| x | d | raw visual feature ของภาพ | backbone |
| X_u | N × d | raw features ใน support set ของ user u | support images |
| h | d → 7 | shared emotion mediator | training-group images |
| z = h(x) | 7 | predicted CORE7 emotions | frozen Stage 1 |
| Z_u | N × 7 | standardized emotion features | training-group scaler |
| y_u | N | personal aesthetic ratings | user support |
| y_pop(x) | 1 | raw-feature population prediction | training-group images |
| w_pop | 7 | population emotion-to-rating coefficients | pooled training group |
| delta_u | 7 | personal residual emotion weights | user support |
| b_u | 1 | personal intercept | user support |
| w_total,u | 7 | w_pop + delta_u, emotion-space reading | derived |
| y_hat_u | 1 | final personal prediction | executed model |

หลักการตั้งชื่อ:

- ชื่อควรบอก stage และ scope เช่น stage1_coef, pop_raw_pred, personal_delta
- อย่าใช้คำว่า weight เฉย ๆ เมื่อมีทั้ง Stage 1, population Stage 2 และ personal Stage 2
- เขียน shape ใน docstring ของจุดที่แปลง representation
- ระบุ scale ด้วย เช่น raw emotion, standardized z และ original rating scale
- เครื่องหมาย hat หมายถึงค่าที่โมเดลประมาณหรือทำนาย ไม่ได้หมายถึงตัวแปรคนละชนิดโดยอัตโนมัติ

---

## 4. การตั้งคำถามวิจัยแบบที่นำไปทดสอบได้

### 4.1 เริ่มจาก tension ไม่ใช่จากโมเดล

คำถามที่ดีมักเกิดจากสิ่งสองอย่างที่ต้องการพร้อมกันแต่ขัดกัน:

- ความแม่นยำกับความอ่านได้
- personalization กับ data efficiency
- semantic meaning กับ representational capacity
- explanation plausibility กับ model faithfulness

งานนี้ไม่ได้เริ่มจาก “จะสร้าง Ridge สอง stage” แต่เริ่มจาก:

> ถ้าบังคับให้ personal model พูดภาษาที่มนุษย์อ่านได้ เราเสีย accuracy เท่าไร และสิ่งที่มันพูดเชื่อถือได้หรือไม่

โมเดลจึงเป็นเครื่องมือสำหรับตอบคำถาม ไม่ใช่จุดหมายของเปเปอร์

### 4.2 เปลี่ยนคำถามกว้างให้เป็น estimand

คำว่า “ดีขึ้น,” “อธิบายได้,” และ “faithful” กว้างเกินไป ต้องแตกเป็นค่าที่วัดได้

ตัวอย่าง:

| คำถามกว้าง | Estimand ที่ตรวจได้ |
|---|---|
| bottleneck เสีย accuracy เท่าไร | mean paired SROCC difference: Hybrid − Direct |
| personalization ช่วยหรือไม่ | Hybrid − Population |
| เสียเพราะ 7 มิติหรือเพราะชื่อ emotion | Hybrid − PCA7 และ Hybrid − Random7 |
| สูตรเฉพาะคนหรือไม่ | Own − peer formula และ Own − population-mean formula |
| concept สำคัญตาม weight หรือไม่ | Full − ablated score ตามอันดับ absolute weight |
| emotion input เฉพาะภาพหรือไม่ | Correct image − cyclic swapped image |
| weight สอดคล้องกับข้อมูลเจ้าของหรือไม่ | correlation(weight, empirical emotion-rating association) |

ก่อนรัน experiment ให้เขียนสี่อย่าง:

1. Intervention คืออะไร
2. Reference/control คืออะไร
3. Unit ที่จับคู่คืออะไร
4. Claim ที่อนุญาตถ้าผลเป็นบวกและถ้าผลเป็นลบคืออะไร

### 4.3 RQ matrix ของงานนี้

| RQ | Intervention | Control | Unit | คำตอบที่อนุญาต |
|---|---|---|---|---|
| RQ1 cost | Hybrid | Direct, Population | user-domain | semantic personal correction trades accuracy for width/readability เท่าไร |
| RQ2 compression | Hybrid 7-d | PCA7, Random7 | user-domain | difference มาจาก width หรือ basis content |
| RQ3 owner | Own formula | population mean, K=5 peers | same evaluation images | formula contains owner-specific information |
| RQ3 weight | hold concept at support mean | Full prediction | same unit | formula-level prediction is sensitive to selected coordinate |
| RQ3 image | correct z_i | all cyclic z_j, no-image zbar | same target y and population term | alignment between image and emotion vector matters |
| RQ3 association | own weights | own and peer empirical associations | user-domain | weights align with owner data beyond shared direction |

### 4.4 Negative results ที่มีคุณค่า

ผลลบไม่ได้ทำให้งานล้มถ้ามันช่วยระบุ bottleneck หรือขอบเขต:

- MLP ต่ำกว่า Ridge แปลว่าความไม่เชิงเส้นที่ทดลองไม่ได้ช่วยภายใต้ข้อมูลต่อคนน้อย
- 14-d mean+SD ไม่ช่วย แปลว่าเพิ่ม spread summary ยังไม่แก้ปัญหา
- PCA ใกล้ Hybrid ภายใต้ anchor แต่แย่กว่าเมื่อ unanchored ช่วยแยกบทบาทของ compression กับ population prior
- GT emotions สูงมาก แปลว่า downstream linear capacity เพียงพอและ Stage 1 perception เป็นข้อจำกัดสำคัญ
- Fashion ต่ำทั้ง Stage 1 และ downstream สนับสนุน diagnostic เดียวกัน

อย่าบังคับผลให้ตรง hypothesis เดิม ให้ถามว่า “ผลนี้ตัด explanation ทางเลือกใดออกได้”

---

## 5. Data design: ความเข้มงวดเริ่มก่อน training

### 5.1 Dataset fact sheet

XPASS-Vis ใน protocol นี้มี:

- 129 participants
- 6,526 images
- 87,836 interactions
- Artwork: 2,345 images, 31,543 interactions
- Fashion: 2,082 images, 28,154 interactions
- Landscape: 2,099 images, 28,139 interactions

แต่ละ interaction มี:

- overall aesthetic rating
- 9 aesthetic-emotion items

ใช้ CORE7 โดยตัด like และ beautiful เพราะ semantic overlap กับ dependent variable อาจทำให้ concept bottleneck กลายเป็นการส่ง label ซ้ำมากกว่าการอธิบาย

Emotion ในไฟล์เป็น 0–4 และ loader ขยับเป็น 1–5 ส่วน overall เก็บ 0–6 ภายใน การบวกค่าคงที่ไม่เปลี่ยน SROCC หรือ PLCC แต่ต้องอธิบาย scale ให้ชัดเพื่อป้องกันความเข้าใจผิด

### 5.2 First-occurrence protocol

มี repeated user-image pairs 4,509 แถวสำหรับ test–retest งานหลักใช้เฉพาะ occurrence แรก เหลือ 83,327 ratings และสงวน repetition ไว้เป็น reliability reference

หลักทั่วไป:

- กำหนดกฎ deduplication ก่อน split
- ให้ deterministic
- อย่าใช้ repeated observation ทั้ง training และ test โดยไม่รู้ตัว
- เก็บ reliability data แยกจาก model evaluation
- อธิบายว่า “first” อิง order ใด

### 5.3 Split ต้องป้องกัน leakage ตามหน่วยที่ shared component เรียนรู้

ผู้ใช้ 129 คนถูกแบ่งเป็น 10 fixed groups แล้วหมุนเป็น 5 folds:

- 7 groups train
- 1 group validation
- 2 groups test

ทุกคนเป็น test user เพียงหนึ่ง fold และ train/validation/test users ไม่ทับกัน

นอกจาก user disjointness ยังต้องตรวจ image disjointness เพราะ Stage 1 และ population model เรียนจากภาพ ถ้าภาพเดียวกันปรากฏใน train กับ test แม้คนละ user ก็อาจเกิด image leakage

กฎ:

> Split ตามเส้นทางที่ข้อมูลสามารถไหลเข้า shared component ไม่ใช่ดูเพียง label ที่อยู่ท้าย pipeline

สำหรับ fine-tuned backbone ต้องใช้ feature file ราย fold เพราะ backbone ที่ fine-tune ด้วยคนทุกคนจะรั่วข้อมูลแม้ downstream split ถูกต้อง

### 5.4 Support/evaluation split

ภายในแต่ละ user-domain:

- ใช้ RandomState(42 + user_id)
- hold out 50 images สำหรับ evaluation ก่อน
- support budget N = 10, 25, 50, 100 มาจาก pool ที่เหลือ
- budget ทั้งหมดใช้ evaluation images ชุดเดียวกัน
- support sets เป็น prefixes ของ shuffled pool เดียวกัน จึงเปรียบเทียบ learning curve ได้ตรงกว่าใช้คนละ draw

เหตุผล:

- test set ต้องไม่เปลี่ยนตาม N
- ความต่างระหว่าง budget ไม่ควรปนกับความยากง่ายของภาพที่สุ่มมาใหม่
- paired comparison ต้องใช้ image set เดียวกันระหว่าง methods

ข้อจำกัดที่ควรยอมรับ:

- uncertainty ที่รายงานเป็น conditional on one deterministic support draw
- ไม่ได้ครอบคลุมความแปรปรวนจากการสุ่ม support images ชุดใหม่
- ถ้าคำถามวิจัยเน้น support-set robustness ต้องเพิ่ม repeated support resampling เป็น experiment ใหม่

### 5.5 Data audit ที่ต้องรัน

อย่างน้อยต้องตรวจ:

- จำนวน user, image, interaction ต่อ domain
- user IDs ไม่ทับระหว่าง train/val/test ใน fold เดียวกัน
- image IDs ไม่ทับตาม protocol
- ทุก user เป็น test เพียงหนึ่ง fold
- ทุก user-domain มี support และ evaluation ครบ
- feature IDs ครอบคลุม rating IDs ที่ใช้จริง
- per-fold feature ถูกโหลดตาม fold
- scale และ missing-value handling เหมือนกันทุก experiment
- paired methods ใช้ unit keys และ evaluation images ตรงกัน

อย่าตรวจ split files ต่อกันอย่างเดียว ต้องตรวจกับ rows ที่ loader ใช้จริงด้วย

---

## 6. Model design: เริ่มจาก causal story ของ computation

### 6.1 Three roles ที่ต้องแยก

1. **Shared perception**
   - เรียน mapping จากภาพไป population-mean emotions
   - ใช้ training-group images
   - freeze ก่อน personalization

2. **Population prediction**
   - เรียน population-level aesthetic score
   - เป็น reference และ anchor
   - ใน Anchor C มาจาก raw visual features

3. **Personal appraisal**
   - เรียนจาก support ratings ของ user
   - ใช้ seven predicted emotions
   - ทำหน้าที่แก้ prediction ตามรสนิยมส่วนบุคคล

การแยกสามส่วนนี้ช่วยตอบ:

- ความผิดพลาดมาจาก perception หรือ appraisal
- prior ช่วยตรงไหน
- parameter ใดเป็นของบุคคลจริง
- explanation อธิบาย full model หรือเพียง personal correction

### 6.2 ทำไมใช้ linear personal head

Linear head ไม่ใช่เพียงเพราะง่าย แต่เพราะมันทำให้:

- น้ำหนักแต่ละ concept อ่านได้
- จำนวน personal parameters ชัดเจน
- ablation และ formula swap ทำได้โดย arithmetic
- fit ได้จาก rating budget เล็ก
- hypothesis space ตรวจสอบได้

ความเรียบง่ายเป็น scientific control: เมื่อ personal head เป็น linear ผลต่างที่เห็นโยงกลับไปที่ representation ได้ง่ายกว่าระบบ end-to-end ขนาดใหญ่

### 6.3 ทำไมต้องมี intercept

ผู้ใช้แต่ละคนใช้ rating scale ต่างกัน บางคนให้คะแนนสูงหรือต่ำเป็นนิสัย แม้น้ำหนัก concept เหมือนกัน

ดังนั้น b_u:

- จับ mean-level rating tendency
- ควรเป็น free, unpenalized per-user intercept
- ไม่ควรถูกนับเป็น semantic concept
- ทำให้คำว่า “everything personal is seven weights” ต้องระวัง เพราะยังมี personal intercept หนึ่งค่า

ถ้าพื้นที่จำกัด เขียนได้ว่า “the feature-dependent personalized component contains seven named coefficients” ซึ่งแม่นกว่า

### 6.4 Variant plain, A, B และ C

#### Plain

    y_hat_u = b_u + z^T w_u

Ridge shrink w_u เข้าหาศูนย์ เมื่อ alpha ใหญ่มาก prediction จึงเข้าใกล้ intercept-only model ไม่ได้เข้าใกล้ Population

นี่เป็นปัญหาใน low-data regime เพราะ prior ที่ดีถูกทิ้งไป

#### A: population prediction เป็น feature ที่แปด

    y_hat_u = b_u + a_u y_pop(x) + z^T w_u

ข้อดี:

- population model อยู่ใน hypothesis space
- a_u อ่านว่า user ตาม consensus มากเพียงใด
- ถ้า a_u = 1 และ w_u = 0 จะได้ population prediction บวก intercept

ข้อควรระวัง:

- ถ้าให้ feature นี้เฉพาะ Hybrid แต่ไม่ให้ Direct การปรับดีขึ้นอาจมาจาก prior ไม่ใช่ bottleneck
- ต้องกำหนด prior ของ a_u และ emotion weights ให้ชัดถ้าต้องการ exact low-data limit

#### B: shrink emotion weights เข้าหา population emotion weights

หา w_pop จาก pooled training group บน standardized emotion representation แล้ว fit:

    delta_u = Ridge(Z_u, y_u - Z_u w_pop - b_pop)
    w_u = w_pop + delta_u
    y_hat_u = Z_u w_u + b_pop + b_u

เมื่อ regularization แรง:

- delta_u เข้าใกล้ 0
- w_u เข้าใกล้ w_pop
- โมเดลกลับไปหา population formula ใน emotion space

นี่เป็น weight-space prior และ population prediction ของ B ผ่าน emotion representation

#### C: personalize residual ของ raw-feature population model

    r_u = y_u - y_pop(X_u)
    delta_u = Ridge(Z_u, r_u)
    y_hat_u = y_pop(x) + b_u + z^T delta_u

เมื่อ regularization แรง:

- delta_u เข้าใกล้ 0
- feature-dependent personal correction หายไป
- เหลือ raw-feature population prediction กับ per-user intercept

C ต่างจาก B ตรง reference function:

- B อ้างอิง Z w_pop + b_pop
- C อ้างอิง y_pop(x) จาก raw visual features
- ทั้งคู่ fit residual แต่ residual คนละตัว

ห้ามสรุปว่า B และ C เหมือนกันเพียงเพราะมีคำว่า residual

### 6.5 Standardization เป็นส่วนหนึ่งของสมการ

ถ้า w_pop ถูก fit ในหน่วยของ training-group scaler แต่ personal user ใช้ scaler ของตัวเอง:

- coefficient อยู่คนละ coordinate system
- shrink toward w_pop ไม่มีความหมาย
- น้ำหนักเทียบกันไม่ได้

ดังนั้น B/C fix standardization statistics จาก training group และใช้กับทุก user ใน fold/domain เดียวกัน

หลักทั่วไป:

> การเปลี่ยน scaler เปลี่ยนความหมายของ coefficient แม้ prediction อาจดูคล้ายเดิม

### 6.6 Fairness ของ baseline

เมื่อเพิ่มความได้เปรียบเชิงโครงสร้างให้ method หลัก ต้องถามว่า baseline ควรได้ treatment เทียบเท่าหรือไม่

ตัวอย่าง:

- Hybrid ได้ population prior → Direct ควรได้ equivalent prior
- Hybrid ใช้ per-fold fine-tuned feature → baseline ที่อ้าง paired comparison ต้องใช้ image subset เดียวกัน
- MLP และ Ridge เทียบ model family → preprocessing และ selection protocol ต้องเหมือนกัน
- 7-d emotion เทียบ PCA/Random → controls ต้องมี width เท่ากัน

กฎ:

> เปลี่ยนเพียง factor ที่ RQ ต้องการทดสอบ สิ่งอื่นตรึงให้เท่ากัน

---

## 7. Representation controls ที่ตอบคำถามได้จริง

### 7.1 Identity / Direct

ใช้ raw backbone feature เป็น personal input ตอบว่า semantic bottleneck เสียอะไรเทียบกับ unrestricted representation

### 7.2 Emotion

7-d named mediator ตอบคำถามหลักด้าน interpretability

### 7.3 PCA7

same-width unsupervised control:

- width เท่ากับ emotion
- content-dependent
- ไม่มี semantic naming

Hybrid − PCA แยก semantic basis ออกจาก compression ได้ดีกว่าเทียบ Direct อย่างเดียว

### 7.4 Random7

content-free หรือ random projection control ต้องระบุ implementation ให้ชัด เพราะ “random” อาจหมายถึง:

- random projection ของ image features ซึ่งยังมี content
- random values ไม่ขึ้นกับ image ซึ่ง content-free
- shuffled mediator predictions ซึ่งคง distribution แต่ทำลาย alignment

ชื่อ table row ต้องไม่ทำให้ reviewer เข้าใจ control ผิด

### 7.5 Shuffled prediction

คง marginal distribution ของ prediction แต่ทำลาย image alignment ใช้ถามว่า performance มาจากการจับคู่ภาพถูกหรือเพียง scale/distribution

### 7.6 Distribution-valued mediators

- 7-d: mean emotion ต่อ concept
- 14-d: mean + across-rater SD
- 35-d: 7 emotions × 5 histogram bins

ต้องมี width-matched controls ถ้าจะสรุปผลด้าน semantics เพราะ 35-d อาจดีขึ้นเพียงเพราะมี capacity มากกว่า

เมื่อผลเพิ่มเล็กน้อย:

- อย่าเขียนว่า distribution ไม่สำคัญโดยเด็ดขาด
- เขียนว่า variants ที่ประเมินไม่ได้ให้ gain คุ้มกับ width ภายใต้ protocol นี้

---

## 8. Loss, optimization และ hyperparameter selection

### 8.1 แยก shared selection กับ personal selection

Shared components เช่น Stage 1 และ population model:

- เลือกด้วย validation-group image-level MSE
- เพราะ target อยู่ระดับภาพและเป็น supervised regression

Personal heads:

- fit candidate แยกใน support set ของ validation user ทุกคน
- score บน disjoint 50-image evaluation set ของ validation user คนนั้น
- ใช้ mean validation-user SROCC เลือก candidate
- freeze หนึ่ง candidate ต่อ fold × domain × condition × budget
- นำ candidate เดียวไปใช้กับ test users ทุกคน

นี่เรียกว่า validation mirrors test protocol

### 8.2 สิ่งที่ห้ามทำ

- เลือก alpha แยกต่อ test user จาก test evaluation score
- ดู test table แล้วขยาย grid เฉพาะ condition ที่อยากให้ดี
- fit candidate และ score บน support rows เดียวกัน
- เลือกด้วย pooled image-level score เมื่อ final estimand อยู่ระดับ user
- report best test seed

### 8.3 Ridge

Penalty grid ของโปรเจกต์:

    logspace(-2, 6, 17)

Ridge objective แบบแนวคิด:

    minimize ||y - Z w||^2 + alpha ||w||^2

ความหมาย:

- alpha เล็ก: fit data มากขึ้น variance สูง
- alpha ใหญ่: coefficient หดเข้าหา prior ของ model
- plain Ridge prior คือ 0
- B prior คือ w_pop
- C prior คือ zero personal residual รอบ y_pop

คำว่า shrink ไม่ได้แปลว่าไม่เทรน แต่แปลว่า loss ลงโทษการห่างจาก reference

### 8.4 Tie-breaking

เมื่อ validation scores ต่างกันไม่เกิน relative tolerance 1e-9:

- linear penalty เลือก regularization ที่แรงกว่า
- MLP เลือก learning rate เล็กกว่าแล้ว budget สั้นกว่า

เหตุผลคือลดการเลือก model ซับซ้อนจาก numerical noise

Tie rule ต้องกำหนดก่อนเห็น test outcome และบันทึกไว้ใน selection log

### 8.5 MLP

Architecture ที่ประเมิน:

- one hidden layer
- 128 ReLU units
- MSE loss
- Adam
- no weight decay
- fixed 500 epochs
- learning-rate grid 1e-4 ถึง 10

Early stopping ปิดเพราะ N=10 จะเหลือ internal hold-out เพียง 1–2 ตัวอย่าง ทำให้ stopping criterion ไม่น่าเชื่อถือ

Joint Stage 1 ใช้แนวคิด:

    L = 0.5 L_score + 0.5 lambda L_concept + L2

score head อ่านจาก concept bottleneck ไม่ใช่ข้าม bottleneck

คำว่า joint ในงานนี้หมายถึง shared Stage-1 concept/score optimization ไม่ได้หมายถึง end-to-end personal training ทุก user พร้อมกัน

### 8.6 Grid-edge result

Hyperparameter ที่ชนะตรงขอบ grid ไม่ใช่ bug โดยอัตโนมัติ แต่เป็น diagnostic:

- ถ้าข้อสรุปสำคัญขึ้นกับ optimum ที่ขอบ ควรขยาย grid
- ถ้าผลเป็น descriptive และ performance แย่ชัด สามารถรายงานขอบเขตที่ทดลองโดยไม่อ้าง global optimum
- selection log ควรมี edge flag

อย่าพูดว่า “we exhaustively tuned” ถ้ายังมี evidence ว่า optimum อาจอยู่นอก grid

### 8.7 Effective degrees of freedom

สำหรับ Ridge:

    df_eff = trace[ Z (Z^T Z + alpha I)^(-1) Z^T ]

คำนวณได้เสถียรกว่าด้วย singular values:

    sum_j s_j^2 / (s_j^2 + alpha)

ใช้เป็น descriptive complexity diagnostic ไม่ใช่จำนวน parameter ดิบและไม่ควรเอาไปแทน hypothesis test

---

## 9. Evaluation unit, seed และ independence

### 9.1 Seed averaging

สาม seeds 0, 1, 2 เป็น repeated computational runs ของ unit เดียวกัน ไม่ใช่ผู้เข้าร่วมเพิ่ม

ลำดับที่ถูก:

1. group ตาม configuration + fold + domain + user
2. average metric ข้าม seeds ภายใน unit
3. ได้ 129 units ต่อ domain หรือ 387 user-domain rows
4. ทำ mean, SEM และ paired test

ลำดับที่ผิด:

1. นำ 129 × 3 domains × 3 seeds = 1,161 rows เข้า test
2. ถือว่าเป็น independent observations

วิธีผิดทำให้ sample size พองและ p-value เล็กเกินจริง

### 9.2 Pooled Average คืออะไร

Average ของงานนี้ไม่ใช่การถ่วงด้วยจำนวนภาพ แต่เป็น:

    mean ของ 387 seed-averaged user-domain scores

ดังนั้นแต่ละ participant-domain unit มีน้ำหนักเท่ากัน

ข้อดี:

- domain ที่มีภาพมากกว่าไม่ครองค่าเฉลี่ย
- สอดคล้องกับ estimand “average participant-domain performance”

ข้อควรระวัง:

- สาม domains ของคนเดียวกัน correlated
- effective independent sampling units มี 129 participants
- SEM แบบ SD/sqrt(387) และ Wilcoxon บน 387 rows anti-conservative

### 9.3 Domain-specific inference

ต่อหนึ่ง domain:

- หนึ่ง unit ต่อ participant
- n = 129 paired participants
- เปรียบเทียบ methods บนคนและ evaluation set เดียวกัน
- ใช้ two-sided paired Wilcoxon signed-rank

### 9.4 Cluster bootstrap สำหรับ Average

ขั้นตอน:

1. สร้าง participant clusters 129 กลุ่ม
2. แต่ละ cluster ถือสาม domain rows
3. สุ่ม participant IDs 129 ค่าแบบ with replacement
4. นำ rows ทั้งสาม domains ของ participant ที่ถูกสุ่มมาด้วยกัน
5. คำนวณ pooled mean หรือ paired mean difference
6. ทำซ้ำ 10,000 ครั้งด้วย seed 0

ได้:

- observed mean
- bootstrap SEM
- percentile 95% CI
- empirical p-value

ถ้า cluster มีขนาดเท่ากันทุกคน ค่าเฉลี่ยเท่ากับ mean ของ participant means ถ้ามี NaN บาง domain implementation ต้องรักษา pooled finite-row estimand และรายงานจำนวนที่ตก

### 9.5 NaN และ constant predictions

ถ้า model ทำนายค่าคงที่:

- rank ไม่มี variation
- SROCC และ PLCC ไม่มีนิยาม
- ค่าที่ถูกคือ NaN ไม่ใช่ 0

แทน NaN ด้วย 0 จะตีความ “คำนวณไม่ได้” เป็น “ไม่มีความสัมพันธ์” ซึ่งคนละความหมาย

วิธีที่ถูก:

- เก็บ NaN ที่ unit level
- pairwise-filter finite values ก่อน test
- mean/SEM ignore เฉพาะ undefined units
- รายงาน n_used และ n_dropped เมื่อมีผลต่อการอ่านตาราง

### 9.6 SEM ไม่ใช่สูตรเดียวทุกตาราง

SD/sqrt(n) ใช้ได้เมื่อ:

- observations อิสระตาม estimand
- n คือจำนวน independent units จริง

ในงานนี้:

- domain columns: unit SEM จาก 129 participants ใช้สูตรปกติได้
- pooled Average: ใช้ participant-cluster bootstrap SEM
- Stage-1 table ที่ aggregate fold-domain estimates มี unit ต่างจาก model table
- figure ที่เฉลี่ย seed ต้อง average seed ก่อนนับ n

ทุก caption ควรตอบ:

- mean over what
- error bar คือ SD, SEM หรือ CI
- n เท่าไร
- clustering อยู่ระดับใด

---

## 10. Hypothesis testing และ multiple comparisons

### 10.1 Paired test ก่อนเสมอเมื่อจับคู่ได้

Methods ใช้:

- user เดียวกัน
- domain เดียวกัน
- evaluation images เดียวกัน
- support split เดียวกัน

จึงควรทดสอบ difference ภายใน unit ไม่ใช่ independent-sample test

### 10.2 Two-sided และ one-sided

Default ของ model comparisons:

- two-sided เพราะความต่างอาจไปได้สองทิศ

Weight alignment against zero:

- one-sided greater
- ใช้ได้เพราะ hypothesis กำหนดทิศทางล่วงหน้าว่า alignment ควรเป็นบวก
- ต้องประกาศใน Methods และ caption

ห้ามเปลี่ยนเป็น one-sided หลังเห็นผลเพียงเพื่อให้ significant

### 10.3 Holm–Bonferroni

Holm ควบคุม family-wise error rate:

1. เรียง raw p-values จากเล็กไปใหญ่
2. เทียบตัวเล็กสุดกับ alpha/m
3. ตัวถัดไปกับ alpha/(m−1)
4. adjusted p-values ต้อง monotonic

ความสำคัญไม่ได้อยู่ที่สูตรอย่างเดียว แต่อยู่ที่นิยาม family

### 10.4 Family map ของโปรเจกต์

| Block | Family | Size | Planned contrasts |
|---|---|---:|---|
| Main Table 1 | domain/Average × metric | 4 | Hybrid vs Population, Direct, Random, PCA |
| Anchor | Average × metric | 8 | Hybrid vs four mediators รวม anchored/unanchored |
| Backbone | domain/Average × metric | 5 | Hybrid vs Population across five backbones |
| Efficiency | domain/Average × metric | 8 | Hybrid/Direct vs Population across four budgets |
| Faithfulness | domain/Average × metric | 13 | formula 6 + ablation 3 + image 4 |
| Weight alignment | domain/Average × coefficient type | 1 | alignment > 0 |

หลักสำคัญ:

- SROCC และ PLCC เป็นคนละ family
- Artwork, Fashion, Landscape เป็นคนละ domain family
- Average เป็น estimand แยก จึงมี family ของตัวเอง
- correlated panels จาก 387 units เดียวกันต้องอยู่ family เดียว ไม่แยกเพื่อทำ p-value ให้เล็ก
- descriptive tables ไม่มี marker และไม่ควรสร้าง inferential claim

### 10.5 Non-significant ไม่เท่ากับ equivalent

เขียนได้:

- “we found no evidence of a difference”
- “the difference was not statistically detected”
- “the observed difference was small and the CI included zero”

ห้ามเขียนเพียงจาก p > .05:

- “the methods are equal”
- “indistinguishable” ในความหมาย equivalence
- “no difference exists”

ถ้าจะอ้าง equivalence ต้อง:

- กำหนด smallest effect size of interest ก่อนเห็นผล
- ทำ TOST หรือ equivalence test ที่เหมาะสม
- แสดง CI อยู่ภายใน equivalence margin

### 10.6 p-value formatting

- ห้ามเขียน p < 1e-6 ถ้าค่าจริงเป็น 2.9e-6
- ใช้ inequality ที่กว้างกว่าหรือเท่ากับค่าจริง เช่น p < 1e-5
- bootstrap zero tail count จาก 10,000 samples ไม่ใช่ p = 0
- รายงานเป็น p < .001 หรืออย่างน้อย p < 1/10,000 ตาม convention ที่ประกาศ
- table marker ต้องสร้างจาก corrected p-value ไม่ใช่ดู raw p
- ถ้ามี *, **, *** ต้องนิยาม threshold ใน caption

---

## 11. Explanation และ faithfulness: ต้องแยกคำให้ถูก

คำว่า “explanation” มีหลายระดับ:

1. **Architectural interpretability**
   - มี named intermediate variables และ linear head

2. **Computational faithfulness**
   - explanation reconstructs หรือ intervenes on quantities in executed computation

3. **Association/plausibility**
   - weight สอดคล้องกับ correlations ใน observed data

4. **Human usefulness**
   - คนเข้าใจ ตรวจ หรือแก้ model ได้จริง

งานนี้มีหลักฐานระดับ 1–3 บางส่วน แต่ยังไม่มี user study สำหรับระดับ 4 จึงเขียนได้ว่าเป็น prerequisite หรือ first step toward editable systems ไม่ใช่พิสูจน์ว่า users edit ได้ผลแล้ว

### 11.1 Formula swap

เป้าหมาย:

> สูตรนี้เฉพาะเจ้าของหรือเป็นเพียง direction ทั่วไป

Protocol ที่ freeze:

- peer ต้องอยู่ fold/domain เดียวกันเพื่อ share scaler และ population model
- ตัด self ออก
- สุ่ม peer K=5 แบบไม่คืนตัวอย่าง
- RandomState(0)
- reset RNG สำหรับ anchored และ unanchored เพื่อให้ pairing reproducible
- score peers ทีละคน แล้วเฉลี่ย correlation ของ 5 สูตร

Comparisons:

- Own vs population-mean formula
- Own vs Other user
- population mean vs Other user

Claim ที่อนุญาต:

- own formula carries owner-specific predictive information

Claim ที่แรงเกิน:

- weights fully characterize human taste
- formula is causal model of preference

### 11.2 Concept ablation

Reported formula-level attribution:

    w_total = w_pop + delta_u
    y_ablate = y_pred - w_total,j (z_j - zbar_j)

โดย zbar_j คือค่าเฉลี่ย concept ใน support set และ ranking ใช้ absolute w_total

ความหมายที่ถูก:

- ถ้าถือ coordinate หนึ่งของ emotion-space formula ไว้ที่ baseline contribution จะเปลี่ยนเท่าไร
- เป็น formula-level attribution

สิ่งที่ไม่ได้พิสูจน์:

- ไม่ได้แยก raw-feature population branch y_pop(x) ออกเป็น 7 อารมณ์
- ไม่ใช่ causal intervention ต่อทุก computational path ของ full Anchor C

Residual-only exact ablation จะใช้ delta_u:

    y_ablate = y_pred - delta_u,j (z_j - zbar_j)

แต่มันตอบคำถามอีกแบบ:

- emotion ใดสำคัญต่อ personal correction ที่รันจริง

ดังนั้นต้องเลือกระหว่าง w_total กับ delta ตาม estimand ก่อนดูผล ไม่ควรเปลี่ยนเงียบ ๆ หลังเห็นตัวเลข

### 11.3 Image swap

เป้าหมาย:

> personal emotion branch อ่าน emotion vector ของภาพเป้าหมายจริงหรือไม่

Anchor C exact execution:

    y_swap,i = y_pop(x_i) + b_u + delta_u^T z(x_j)

สิ่งที่ตรึง:

- target true rating y_i
- target population prediction y_pop(x_i)
- personal bias
- personal delta

สิ่งที่เปลี่ยน:

- emotion vector จากภาพ i เป็นภาพ j ใน domain เดียวกัน

ใช้ 49 non-zero cyclic shifts บน evaluation set 50 ภาพ:

- ทุกภาพเป้าหมายเห็น emotion vector ของภาพอื่นครบทุกภาพครั้งหนึ่ง
- ไม่มี Monte Carlo uncertainty จากการสุ่ม swap
- row order และ cycle ต้อง deterministic

No-image ใช้ support-set mean zbar แทน image-specific z

Claim ที่อนุญาต:

- correct image-emotion alignment matters to prediction

Claim ที่แรงเกิน:

- wrong emotion input “misleads” model เพราะ swapped ต่ำกว่า no-image

เหตุผลคือ swapped vectors ยังมี variance/covariance structure ต่างจาก constant mean ผล covariance-matched Gaussian control ใกล้ cyclic swap ชี้ว่า scale/variance อธิบายส่วนนี้ได้ จึงควรใช้เป็น post-hoc descriptive scale check ไม่ใช่ core confirmatory experiment

### 11.4 Weight alignment

ต่อหนึ่ง unit:

1. มี weight vector 7 ค่า
2. คำนวณ empirical association ของ predicted emotion แต่ละตัวกับ raw aesthetic ratings ของ user
3. correlate weight vector กับ association vector

Reported quantities:

- Spearman alignment
- Pearson alignment
- positive-unit percentage
- one-sided test against zero

นี่บอกว่า formula weights สอดคล้องกับ pattern ที่เห็นในข้อมูลของเจ้าของ

แต่ shared population direction อาจทำให้ทุกคน align คล้ายกัน จึงมี peer-null:

- จับ weight ของ user หนึ่งกับ empirical associations ของ peer ใน fold/domain เดียวกัน
- ใช้ deterministic cyclic peer mapping
- เปรียบเทียบ own alignment กับ peer alignment
- personal excess คือ own − peer

ค่าประมาณใน release:

- own Spearman alignment ประมาณ .525
- peer-null ประมาณ .453
- personal excess ประมาณ .072 พร้อม CI ที่เป็นบวก

ความหมาย:

- ส่วนใหญ่ของ alignment มี shared direction จริง
- แต่มี owner-specific excess เหนือ shared direction

ไม่ควรซ่อน peer null และไม่ควรเขียนว่า alignment ทั้งหมดเป็น personalization

### 11.5 Qualitative example

ภาพ qualitative ควรทำหน้าที่:

- ทำให้ผู้อ่านเห็นว่าค่า concept และ weight แปลเป็น prediction อย่างไร
- แสดงตัวอย่างสำเร็จและ failure case
- เชื่อมสูตรกับภาพอย่างอ่านได้

ต้องระบุ:

- selection rule เป็น deterministic หรือ post-hoc อย่างไร
- ใช้ total emotion-space formula หรือ executed delta branch
- raw/predicted emotion scale
- contribution คำนวณจาก standardized z หรือ raw emotion
- ภาพได้รับอนุญาตให้เผยแพร่หรือไม่

อย่าใช้ qualitative example เป็นหลักฐานเชิงสถิติ และอย่าเลือกแต่ภาพสวยที่ทำให้โมเดลดูดีโดยไม่เปิดเผย selection criterion

---

## 12. จาก raw run สู่ตัวเลขในเปเปอร์: data lineage

### 12.1 ลำดับที่ควรเป็น

    immutable run outputs
        ↓ validate schema and keys
    canonical unit database
        ↓ seed averaging
    statistical contrasts
        ↓ correction and uncertainty
    table/figure artifacts
        ↓ manuscript audit
    PDF

ทุกลูกศรต้องมีโค้ด ไม่ควรพึ่ง copy-paste ด้วยมือ

### 12.2 Canonical databases ของ release

แหล่งสำคัญ:

- raw_all_final.csv: seed-level model metrics จากทุก configuration
- formula_units_final.csv: formula-level rows ของ anchored/unanchored units
- formula_images_final.csv: 50 held-out image rows ต่อ formula unit
- stage1_accuracy.csv
- upper_bound_units_final.csv
- emotion_importance_units_final.csv
- mlp_architecture_units_final.csv
- reference_bounds_final.csv
- regularizer_units_final.csv
- test_retest_units.csv
- statistics/holm_corrected.csv

ความสัมพันธ์:

- raw_all_final เป็นฐานของ main/backbone/efficiency/model tables
- formula databases เป็นฐานของ faithfulness diagnostics
- specialized unit filesรองรับ analyses ที่ไม่ได้อยู่ใน raw_all schema
- holm_corrected เป็น derived audit artifact ไม่ใช่ raw evidence

อย่าฝืนรวมทุกอย่างเป็น CSV เดียวถ้า unit หรือ schema ต่างกัน เพราะการรวมแบบนั้นซ่อนความหมายของแต่ละ row

### 12.3 Source manifest

Manifest ที่ดีบันทึก:

- source path แบบ relative
- row count
- schema/version
- checksum SHA-256
- priority เมื่อมี source ซ้ำ
- timestamp หรือ commit ของ generation code

Manifest ไม่ใช่สิ่งจำเป็นต่อโมเดล แต่สำคัญต่อ provenance:

- รู้ว่า canonical DB สร้างจากไฟล์ใด
- ตรวจ accidental replacement
- audit ว่า source 22 files ถูกใช้ครบ

ถ้าไม่แจก manifest ห้ามเขียนใน supplementary ว่า reviewer ตรวจ checksum จาก manifest ได้

### 12.4 Duplicate policy

เมื่อ merge runs:

- same key และ metrics เท่ากันถึง precision ที่กำหนด → no-op
- same key แต่ metrics ขัดกัน → ห้าม overwrite เงียบ
- เก็บ conflict ทั้งคู่หรือหยุดด้วย error พร้อม source paths

Key ต้องรวมสิ่งที่ทำให้ run แตกต่างจริง เช่น:

- experiment
- backbone
- mediator
- head
- variant
- n_train
- fold
- domain
- user_id
- seed

### 12.5 Freeze policy

freeze ไม่ได้แปลว่า “ห้ามแก้ไฟล์ตลอดไป” แต่แปลว่า:

- ระบุเวอร์ชันที่ใช้ใน submission
- downstream regeneration ห้ามทับ frozen outputs
- rerun เขียนไป isolated directory
- เปรียบเทียบใหม่กับ frozen version
- ถ้าต้องเปลี่ยน methodology ให้สร้าง version ใหม่และบันทึก rationale

หลัง submission ห้ามเปลี่ยน estimator หรือ protocol แล้วแอบแทนตัวเลขเดิมโดยไม่ version

---

## 13. Reproducibility ที่ reviewer ใช้ได้จริง

### 13.1 สองระดับของ reproduction

#### Downstream exact reproduction

ใช้ precomputed unit-level outputs เพื่อ regenerate:

- statistics
- tables
- figures
- faithfulness diagnostics

ควรรันได้เร็วและไม่ต้องมี licensed raw dataset หรือ huge features

#### Full model reproduction

เริ่มจาก:

- licensed ratings
- extracted/fine-tuned features
- fixed splits
- training scripts

ใช้เวลานานและอาจต้อง hardware เฉพาะ

README ต้องแยกสองระดับนี้ตรง ๆ อย่าเขียนว่า “reproduce all experiments in seconds” ถ้าความจริงเป็นเพียง downstream regeneration

### 13.2 Clean-room command

มาตรฐานที่ดีของ release:

    uv sync --frozen
    uv run python reproduce.py --output-dir tmp/reviewer_check

ผลที่ต้องเกิด:

- อ่าน frozen audit inputs
- สร้าง outputs ใหม่ใน tmp/reviewer_check
- regenerate statistics/tables/figures
- compare frozen statistics row-for-row
- ไม่แก้ output/ เดิม
- จบด้วย non-zero exit code เมื่อ mismatch

หลัง command สำเร็จ tmp directory ลบได้ แต่ reproduce.py ต้องอยู่ เพราะมันคือ entry point ของ reviewer

### 13.3 Environment

ใช้:

- pyproject.toml สำหรับ declared dependencies
- uv.lock เป็น executable reference environment
- Python version และ library versions ใน audit report

Cross-platform:

- BLAS และ floating-point summation order อาจทำให้ค่าแตกต่างระดับเล็ก
- same locked environment และ single-thread BLAS ช่วยลดความต่าง
- กำหนด tolerance ที่เล็กกว่าความละเอียดของค่าที่รายงาน
- อย่าอ้าง byte-identical cross-platform ถ้ายังไม่ได้ทดสอบจริง

### 13.4 Randomness

ทุก RNG ต้องมี:

- purpose
- seed
- scope
- call order

ตัวอย่างในโปรเจกต์:

- support split: 42 + user_id
- run-level mediator: fold + seed offset
- per-user MLP: user_id + run seed offset
- population MLP: 100 + fold
- PCA: seed 0
- formula swap: RandomState(0)
- cluster bootstrap: seed 0

ระวังว่า RNG เดียวกันแต่เพิ่ม random draw ก่อนหน้าอาจเปลี่ยนผลทั้งหมด ควรแยก generator ตาม purpose หรือบันทึก call order

### 13.5 Release hygiene

ไม่ควรใส่:

- .git
- .venv
- __pycache__
- tmp
- IDE state
- local cache
- backup/no-use folders
- absolute personal paths
- names/emails ใน metadata, comments หรือ README สำหรับ double blind

ควรใส่:

- code ที่ entry point เรียกจริง
- locked environment
- fixed splits
- compact audit CSVs
- schema/data documentation
- one-command reproduction
- license/data-access note

Comments ใน code ควรอธิบาย:

- invariant
- reason for a design choice
- shape/unit
- leakage prevention
- statistical scope

ไม่ควรเป็น:

- บทสนทนากับตัวเอง
- การแก้ตัว
- ประวัติว่าครั้งก่อนทำพลาดอะไร
- claim ที่ไม่ตรงกับ implementation ปัจจุบัน

---

## 14. Code architecture ที่พอดี ไม่ over-engineer

### 14.1 แบ่งตาม responsibility ไม่ใช่ตามชื่อ experiment อย่างเดียว

โครงที่สมเหตุผล:

- data/: loader, aggregation, split
- modeling/: backbone adapter, mediator, head, pipeline
- experiments/: orchestration และ scientific intervention
- utils/: metrics, statistics, result IO, verification
- reproduce.py: release-level orchestrator

ชื่อซ้ำอย่าง modeling/backbones.py กับ experiments/backbone.py ไม่ได้ซ้ำหน้าที่:

- modeling/backbones.py โหลด feature representation
- experiments/backbone.py รัน comparison across backbones

แต่ต้องมี README tree อธิบาย ไม่เช่นนั้น reviewer จะงง

### 14.2 เมื่อใดควรสร้าง abstraction

สร้าง shared function/class เมื่อ:

- logic เดียวกันใช้ 3 จุดขึ้นไป
- invariant สำคัญต้องบังคับจากศูนย์กลาง
- duplication เสี่ยงให้สถิติ/format ต่างกัน

อย่าสร้างเมื่อ:

- wrapper เพียงเรียกสองฟังก์ชันหนึ่งครั้ง
- ทำให้ call chain ยาวโดยไม่ลดความเสี่ยง
- reviewer ต้องเปิดห้าไฟล์เพื่อเข้าใจ equation เดียว

เป้าหมายไม่ใช่จำนวนไฟล์น้อยที่สุด แต่เป็นระยะทางสั้นที่สุดจากคำถามถึง implementation

### 14.3 Entry point

main.py ควร:

- expose experiments ที่ผู้ใช้รันจริง
- parse config
- wire dataset/backbone/split/pipeline
- ส่ง output ไป directory ที่ระบุ
- เขียน config ของ run

reproduce.py ควร:

- ใช้เฉพาะ released artifacts
- ไม่ฝึก model ใหม่โดยไม่ตั้งใจ
- สร้าง derived outputs แบบ isolated
- verify กับ frozen reference

สอง entry points ตอบคนละ use case จึงควรมีทั้งคู่

### 14.4 Assertions ที่มีคุณค่า

ควร assert:

- expected unit counts
- uniqueness ของ keys
- seed set
- 50 images ต่อ formula unit
- anchored/unanchored image keys ตรงกัน
- true labels ตรงกัน
- selected hyperparameter อยู่ใน grid
- every reported contrast has aligned pairs
- Holm family size ตรง definition

Assertion ทำหน้าที่เปลี่ยน silent corruption เป็น explicit failure

### 14.5 Selection log

ต่อ selection event ควรเก็บ:

- stage/component
- fold/domain/condition/budget
- candidate grid
- score ของทุก candidate
- winner
- tie rule
- edge flag
- n validation units
- metric และ direction

ถ้ามีเพียง selected alpha อย่างเดียว reviewer ตรวจ selection bias ไม่ได้

---

## 15. Debugging และ audit แบบนักวิจัย

### 15.1 แยกสามประเภทของปัญหา

1. **Code bug**
   - implementation ไม่ตรง protocol
   - เช่น seed rows ถูกนับเป็น independent

2. **Method choice**
   - implementation ตรงที่กำหนด แต่มีทางเลือกที่ถกเถียงได้
   - เช่นใช้ one fixed support draw

3. **Wording mismatch**
   - code/result ถูก แต่คำอธิบายเกิน scope
   - เช่นเรียก formula-level attribution ว่า exact full-model decomposition

อย่า rerun training ถ้าปัญหาเป็น wording และอย่าแก้ wordingกลบ code bug

### 15.2 เมื่อเลขไม่ตรง

ตรวจตามลำดับ:

1. ตัวเลขใน PDF
2. source table tex
3. generated table
4. summary CSV
5. corrected statistics CSV
6. unit-level input
7. seed-level raw database
8. immutable source run
9. code/config/commit ที่สร้าง source

ถามทุกขั้น:

- row key เดียวกันหรือไม่
- anchored/unanchored ตรงหรือไม่
- K=5 หรือ cyclic
- seed-averaged หรือ seed-level
- domain หรือ pooled Average
- SD หรือ SEM
- raw หรือ Holm p
- old file หรือ current frozen file

### 15.3 Targeted rerun

อย่าเริ่ม retrain หลายแสน rows เพราะ summary หนึ่งช่องผิด

ลำดับที่ปลอดภัย:

- reproduce downstream ใน temp
- diff row-for-row
- rerunเฉพาะ contrast หรือ unit ที่ mismatch
- rerunเฉพาะ experiment/model slice ที่จำเป็น
- retrain full grid เมื่อมีหลักฐานว่า source run ผิด

### 15.4 Numerical tolerance

ใช้:

- full precision ใน CSV
- tolerance ตามชนิด metric
- rounding เฉพาะ presentation

การ compare ด้วยค่าปัดสามตำแหน่งอาจซ่อน regression ส่วน compare exact bytes ข้าม BLAS อาจเข้มเกินไป ต้องเลือกระดับที่ตอบความเสี่ยงจริง

### 15.5 Definition of done

Experiment ยังไม่เสร็จเมื่อ training จบ แต่เสร็จเมื่อ:

- unit counts ผ่าน
- keys unique/aligned
- selection logs ครบ
- raw outputs immutable
- derived stats regenerate ได้
- table/figure regenerate ได้
- numbers diff กับ manuscript ผ่าน
- caption ระบุ unit/uncertainty/test
- claim ตรง estimand
- release clean-room ผ่าน

---

## 16. การออกแบบกราฟและตาราง

### 16.1 หน้าที่ของ figure กับ table

Figure เหมาะกับ:

- pattern
- trend
- comparison structure
- effect direction
- interaction across budget/domain

Table เหมาะกับ:

- exact values
- many metrics
- reproducibility
- detailed supplementary audit

ถ้า figure และ table บอกข้อมูลเหมือนกันทุกอย่าง ให้เลือกตัวหลักหนึ่งอย่างและย้ายรายละเอียดอีกแบบไป supplementary หรือ release CSV

### 16.2 Significance annotation แบบภาพตัวอย่าง

ภาพตัวอย่างจาก Hayashi-san มีหลักที่ดี:

- significance bracket อยู่ตรงคู่ที่เปรียบเทียบ
- label n.s., *, **, *** อ่านได้ทันที
- condition order และสีเหมือนกันทุก panel
- panels ใช้ shared y-scale
- title ของแต่ละ outcome อยู่เหนือ panel
- legend บอกชื่อกลุ่มและ sample size
- uncertainty/distribution แสดงโดย graphical layer ที่แยกกัน

นำมาใช้กับ figures ของงานนี้โดย:

1. กำหนด reference comparison ใน caption
2. วาง bracket เฉพาะ contrasts ที่อยู่ใน pre-specified Holm family
3. ใช้ corrected p-value
4. กำหนด threshold เช่น:
   - n.s.: p_adj ≥ .05
   - *: p_adj < .05
   - **: p_adj < .01
   - ***: p_adj < .001
5. ห้าม marker ชน title หรือถูก crop
6. รักษา vertical spacing ของ nested brackets
7. ใช้สีที่ color-blind friendly และยังแยกได้เมื่อพิมพ์ grayscale
8. caption ต้องบอก unit, seed averaging, error bar และ correction family

ถ้ามี comparisons มากจน bracket รก:

- แสดงเฉพาะ confirmatory reference contrasts
- ใส่ exact corrected p-values ใน supplementary table/CSV
- ไม่ตัด contrast ออกเพียงเพราะ non-significant

### 16.3 Error bars

ห้ามใช้ error bar โดยไม่บอกว่าอะไร:

- SD = spread ของ units
- SEM = uncertainty ของ mean
- 95% CI = interval estimate

สำหรับ Average ของงานนี้ควรใช้ participant-cluster bootstrap SEM/CI ไม่ใช่ naive 387-row SEM

### 16.4 Table rules

- decimals เท่ากันใน column เดียวกัน
- leading zero policy คงที่ตาม venue
- bold = best value ไม่ใช่ significance
- marker = inferential comparison ที่ caption นิยาม
- row order สอดคล้องกับ narrative
- table note อธิบาย notation/protocol
- ผลการตีความควรอยู่ใน body text ไม่ยัดใน tablenote
- descriptive tables ต้องระบุ descriptive และไม่มี dagger

### 16.5 Rounding

เก็บ full precision แล้ว format ด้วย deterministic rule

ข้อควรระวัง:

- Python formatting มักใช้ round-to-even ไม่ใช่ “0.5 ปัดขึ้น” ทุกกรณี
- อย่าปัด mean ของ rounded unit values
- ค่า difference ในข้อความควรคำนวณจาก full precision แล้วปัด ไม่ลบค่าที่ปัดแล้ว
- table และ text ต้องใช้ formatter หรือ audit script เดียวกัน

### 16.6 Visual QA

ก่อนส่ง:

- render PDF
- ดูทุกหน้า ไม่ดูเฉพาะ LaTeX compile success
- ตรวจ crop, overlap, tiny font, orphan caption, sparse float pages
- เช็ก black-and-white readability
- เช็ก label กับ legend
- เช็ก figure marker ตรง statistics CSV
- เช็ก cross-reference หลังย้าย tables/figures

---

## 17. การเขียนเปเปอร์ให้กระชับ ฉลาด และน่าเชื่อถือ

### 17.1 จังหวะประโยคหลัก

ย่อหน้าผลที่แข็งแรงมักมี:

1. setup/reference
2. observed effect
3. uncertainty/statistical evidence
4. interpretation ที่ไม่เกินผล

ตัวอย่าง:

> Under the population anchor, Hybrid reached .459 SROCC, compared with .439 for Population and .478 for Direct. The paired differences were retained after Holm correction. Thus, the emotion-mediated personal correction improved over the population reference while leaving a measurable gap to unrestricted personalization.

ไม่ต้องใส่ทุกเลขในประโยคเดียว เลือกตัวที่ทำหน้าที่ตอบ RQ

### 17.2 Claim–evidence–scope

ทุก claim ควรมีสามส่วน:

- Claim: เราพบอะไร
- Evidence: comparison/number/test ใดสนับสนุน
- Scope: ภายใต้ dataset/backbone/protocol ใด

ตัวอย่าง:

> On the primary Qwen3-VL 8B backbone, Hybrid and PCA were close under the population anchor; this relation was not uniform across backbones.

ประโยคนี้ดีกว่า:

> Naming the dimensions costs nothing.

เพราะประโยคหลัง generalize เกินหลักฐาน

### 17.3 คำที่ใช้กับระดับหลักฐาน

| หลักฐาน | คำที่เหมาะ | คำที่ควรเลี่ยง |
|---|---|---|
| corrected p < alpha | significantly differed/outperformed | proved universally |
| p ≥ alpha | no evidence of a difference | equivalent/the same |
| observational alignment | aligned/was consistent with | caused/determined |
| matched intervention | supports a mechanistic role | proves human cognition |
| one dataset | in XPASS-Vis/under this protocol | generally/across people |
| post-hoc control | descriptive scale check | confirmatory test |
| qualitative example | illustrates | demonstrates statistically |

### 17.4 Abstract

โครงที่ดี:

1. **Problem:** PIAA models personal taste but hides its logic
2. **Idea:** aesthetic emotions as interpretable personal coordinates
3. **Design:** shared perception + personal linear correction
4. **Protocol:** dataset, folds, controls, primary backbone
5. **Results:** cost relative to Population/Direct
6. **Validation:** owner, weight, image, association diagnostics
7. **Boundary:** Stage 1 limits system
8. **Implication:** prerequisite toward inspectable/correctable systems

Abstract ต้อง:

- ไม่มี citation ตาม venue ถ้าไม่จำเป็น
- ไม่มี claim ที่ไม่อยู่ Results
- ไม่ยัด supplementary-only result หากไม่ได้อธิบายพอ
- ใช้ exact scope ของ computation

ประโยคที่ปลอดภัยกว่า:

> The feature-dependent personalized component is expressed by seven named emotion coefficients.

ประโยคที่เสี่ยง:

> Everything personal is seven weights.

เพราะยังมี intercept และ population branch

### 17.5 Introduction

โครง:

1. ความสำคัญของ task
2. ช่องว่างของวิธีเดิม
3. เหตุผลทางมนุษย์/ทฤษฎีว่าทำไม mediator นี้สมเหตุผล
4. วิธีใช้ model เป็น scientific instrument
5. RQs
6. contributions

สิ่งที่ดีจากงานนี้:

- ไม่ขาย architecture ซับซ้อน
- วางความเรียบง่ายเป็นข้อดีเชิง identification
- ระบุ non-goal ว่า emotions ไม่ได้อธิบายทุกอย่าง
- เชื่อม interpretability กับการที่ผู้ใช้ตรวจและแก้โมเดลของตัวเองได้

สิ่งที่ต้องระวัง:

- ถ้า Anchor C มี raw-feature population branch ห้ามเขียนว่า full prediction must pass through emotions
- “first study” ต้องมี literature search ครอบคลุมพอ
- contribution ต้องผูกกับ experiment ที่มีจริง

### 17.6 Related work

อย่าเขียนเป็นรายการ paper ต่อ paper ให้จัดตามข้อจำกัดที่นำไปสู่ช่องว่าง:

1. GIAA → population average แต่ไม่ personalize
2. PIAA → personalize แต่ black-box
3. interpretable IAA → explain image-side attributes แต่ไม่ personal response
4. affective psychology → supports emotions as response-side concepts
5. concept bottleneck → architectural mechanism
6. gap → interpretable personalized appraisal

สำหรับแต่ละ citation ถาม:

- อ้างเพื่อรองรับ fact, method, theory หรือ contrast
- source ต้นฉบับหรือ review
- claim กว้างกว่าที่ source พูดหรือไม่
- งานล่าสุดเปลี่ยน “first” claim หรือไม่

Technical claims ควรอ้าง primary source:

- dataset paper สำหรับ counts/protocol
- original method paper สำหรับ model
- psychology study สำหรับ human behavior
- official library docs ไม่ใช่ blog เมื่อต้องอธิบาย implementation

### 17.7 Results

Results ควรตอบ RQ ตามลำดับ ไม่ใช่ตามลำดับที่รัน experiment

กฎ:

- เปิดด้วย primary comparison
- รายงาน effect size ก่อนหรือพร้อม p-value
- แยก confirmatory กับ descriptive
- negative result เขียนตามจริง
- ห้าม infer mechanism ที่ experiment ไม่ isolate
- ทุก p inequality ต้องตรวจจาก corrected CSV

รูปแบบ:

> X increased from A to B (difference D, 95% CI [...], Holm-adjusted p = ...).

ถ้าพื้นที่น้อย:

> X exceeded Y across all five backbones (adjusted p ≤ ... for the pooled Average).

ต้องระบุ “pooled Average” ถ้า domain-wise ไม่ครบ

### 17.8 Discussion

Discussion ไม่ใช่ Results รอบสอง ควร:

- รวมหลายผลเป็น insight
- เสนอ mechanism อย่างมี qualifier
- พูด alternative explanation
- กำหนด boundary conditions
- เชื่อมสู่ future work ที่แก้ bottleneck จริง

ตัวอย่าง insight ของงานนี้:

- GT emotions สูงแต่ predicted emotionsต่ำ → perception bottleneck
- anchored PCA ใกล้ Hybrid แต่ unanchored PCA ต่ำ → prior และ semantic basis ทำงานคนละส่วน
- domain pattern ตรง Stage-1 fidelity → representation improvement เป็นทิศทางที่มีเหตุผล

อย่านำ p-value ใหม่หรือ analysis ที่ไม่เคยปรากฏมาใส่ Discussion

### 17.9 Limitations

Limitation ที่ดีมีสามส่วน:

1. ขอบเขตอะไรยังไม่ครอบคลุม
2. เหตุใดไม่ทำลายคำถามหลัก
3. experiment ต่อไปแก้ได้อย่างไร

ตัวอย่าง:

> The seven concepts are fixed by the available dataset. This limits vocabulary coverage but not the within-vocabulary comparison studied here; future data collection can combine aesthetic emotions with objective attributes.

ไม่ต้องเขียนภาษาที่เหมือนประกาศว่า model ใช้ไม่ได้ทั้งหมด

### 17.10 Conclusion

Conclusion ควรแคบกว่า abstract:

- ตอบคำถามที่ตั้ง
- ไม่เพิ่ม new mechanism
- ไม่ยกระดับ “no evidence” เป็น “equal”
- ไม่ generalize primary-backbone result ไปทุก backbone
- จบด้วย implication ที่หลักฐานรองรับ

---

## 18. Writing templates ภาษาอังกฤษ

### 18.1 Methods: split

> Participants were assigned to fixed train, validation, and test groups. Shared components were fitted on training-group images only. Personal-head hyperparameters were selected by reproducing the support/evaluation protocol within validation users and were frozen before application to test users.

### 18.2 Methods: seed

> The three computational seeds were averaged within each user–domain unit before inference. Seeds were therefore not treated as independent observations.

### 18.3 Methods: clustered Average

> Domain-specific inference used one paired observation per participant. For the pooled Average, participants were resampled as clusters, retaining all three domain observations for each selected participant.

### 18.4 Result: significant

> Hybrid exceeded the population reference by .020 SROCC in the pooled Average (95% CI [...], Holm-adjusted p = ...).

### 18.5 Result: non-significant

> The observed Hybrid–PCA difference was .001, and we found no evidence of a difference under the primary anchored setting (95% CI [...], Holm-adjusted p = .761).

### 18.6 Descriptive result

> The evaluated MLP variants performed below Ridge → Ridge and are reported descriptively; no inferential marker is attached to this ablation.

### 18.7 Faithfulness scope

> The ablation measures attribution within the reported emotion-space formula while holding the raw-feature population prediction fixed; it is not a decomposition of the population branch.

### 18.8 Post-hoc control

> As a post-hoc descriptive scale check, covariance-matched Gaussian emotion vectors produced a similar pooled score to the cyclic swap.

### 18.9 Limitation

> This result establishes image alignment within the fitted model, not a causal claim about the emotions experienced by the participant.

### 18.10 Future work

> The gap between measured and predicted emotions points to perception, rather than downstream linear capacity, as the immediate target for improvement.

---

## 19. Reviewer mindset: red-team checklist

ก่อนส่ง ให้อ่านงานเหมือน reviewer ที่ต้องหาสาเหตุ reject

### 19.1 Data and leakage

- participant split disjoint จริงหรือไม่
- image split disjoint จริงหรือไม่
- backbone fine-tuning ใช้ test images/users หรือไม่
- scaler fit ด้วยข้อมูลใด
- first-session filter ใช้ก่อน split หรือไม่
- duplicated images/users ปะปนหรือไม่
- test user ถูกใช้เลือก alpha หรือไม่

### 19.2 Fair comparison

- methods ใช้ support/eval images เดียวกันหรือไม่
- prior ถูกให้กับทุก baseline ที่ควรได้หรือไม่
- width controls เท่ากันหรือไม่
- preprocessing เท่ากันหรือไม่
- compute/training budget แตกต่างจน claim ไม่ยุติธรรมหรือไม่

### 19.3 Statistics

- unit independence ถูกหรือไม่
- seed averaged ก่อน test หรือไม่
- Average cluster participants หรือไม่
- paired test ใช้ alignment keys ถูกหรือไม่
- family definitions เขียนก่อน correction หรือไม่
- correlated blocks ถูกแยก family เพื่อให้ผ่านหรือไม่
- p-value text ตรง CSV หรือไม่
- non-significant ถูกเรียกว่า equivalent หรือไม่
- NaN ถูกแทน 0 หรือไม่

### 19.4 Interpretation

- equation ตรง code หรือไม่
- weight ที่ plot อยู่ใน execution path หรือเป็น derived representation
- ablation ตัดสิ่งที่ claim จริงหรือไม่
- image swap ตรึง branch ไหน
- association ถูกเขียนเป็น causation หรือไม่
- qualitative example ถูกยกเป็น proof หรือไม่
- result บน backbone เดียวถูก generalize หรือไม่

### 19.5 Reproducibility

- reviewer รัน command จาก clean checkout ได้หรือไม่
- release มี missing module หรือไม่
- output เขียนทับ frozen data หรือไม่
- README บอก data availability ตรงไฟล์จริงหรือไม่
- absolute path/name/metadata ทำลาย anonymity หรือไม่
- table generator สร้างเลขตรง PDF หรือไม่
- old files อาจถูกดึงผิดหรือไม่

### 19.6 Writing

- title/abstract/conclusion พูด claim เดียวกันหรือไม่
- contributions มี evidence ใน Results หรือไม่
- caption นิยาม marker และ error bar หรือไม่
- cross-references ถูกหลังย้าย supplementary หรือไม่
- related-work “first” claim ยังจริงหรือไม่
- limitations ตรงและไม่ self-sabotage หรือไม่

---

## 20. วิธีรับมือเมื่อพบข้อผิดพลาดใกล้ deadline

### 20.1 Triage ตามผลกระทบ

#### ระดับแดง: ต้องหยุดและแก้

- leakage
- test-set model selection
- wrong outcome/label
- incorrect unit independence
- code ไม่ตรง equation หลัก
- main result มาจาก file/version ผิด

#### ระดับเหลือง: แก้สถิติหรือถ้อยคำ

- SEM ใช้ n ผิดแต่ means ถูก
- raw p ยังไม่ corrected
- non-significant overclaim
- full-model claim จาก branch-level diagnostic
- caption ไม่ประกาศ one-sided test

#### ระดับเขียว: presentation

- wording polish
- table spacing
- comment cleanup
- cross-reference
- decimal formatting

### 20.2 อย่าปกปิด

ถ้าพบปัญหาหลัง submission:

- เก็บ evidence
- ระบุว่ากระทบ mean, uncertainty, significance หรือ conclusion
- แก้ release/arXiv/rebuttal ตามกฎ venue
- บอก coauthors อย่างสั้นและตรง

การแก้ methodological error อย่างโปร่งใสปลอดภัยกว่าปล่อยให้ reviewer พบเอง

### 20.3 อย่าตื่นตระหนกจาก wording mismatch

ถ้า computation ถูกแต่ประโยคแรงเกิน:

- ไม่ต้อง retrain
- เปลี่ยน claim ให้ตรง scope
- เพิ่ม caveat สั้นเฉพาะจุด
- ตรวจ abstract, results, discussion, conclusion ให้สอดคล้อง

---

## 21. Lessons learned จากจุดที่งานนี้เกือบสับสน

### 21.1 386 กับ 387

อย่าอธิบาย count จากอาการ เช่น SD = 0 หรือ non-zero ให้หา unit enumeration โดยตรง

คำตอบที่ถูกต้องต้องมาจาก:

- unique participant IDs
- unique domains
- fold coverage
- missing unit report
- exact commit diff

387 = 129 participants × 3 domains เมื่อแต่ละคนมีครบทุก domain และเป็น test คนละครั้ง

### 21.2 SD เป็นศูนย์

SD = 0 อาจเกิดจาก:

- deterministic row duplicated across seeds
- seed averagingผิดลำดับ
- metric ถูก copy ซ้ำ
- control ไม่ใช้ randomness จริง
- grouping key ลืม field

ห้ามเดาว่า “เพราะ residual training” จนกว่าจะ trace rows และ RNG

### 21.3 w_pop, delta และ w_total

อย่าเลือก weight เพื่อ visualization เพราะชื่อสวยกว่า ให้เลือกตามคำถาม:

- exact personal residual computation → delta
- B full emotion formula → w_pop + delta
- C emotion-space attribution → w_total พร้อม scope disclosure
- raw-feature population prediction → อธิบายด้วย raw branch ไม่ใช่ w_pop

### 21.4 Formula swap version

K=5 peers และ cyclic peer mapping เป็นคนละ estimator:

- K=5 ประเมิน average performance of sampled peers
- cyclic mapping เลือก peer เดียวต่อ unit อย่าง deterministic

เมื่อ freeze K=5:

- code, CSV, table, Holm input และ manuscript ต้องใช้ K=5 เหมือนกัน
- cyclic version ย้ายออกจาก active paths หรือ label ชัดว่า audit/old

### 21.5 SEM .011 กับ .012

ความต่างอาจมาจาก:

- naive unit SEM vs participant-cluster bootstrap
- SD กับ SEM
- seed-level vs seed-averaged
- 387 rows vs 129 clusters
- different source version
- rounding of full precision

วิธีหยุดการแก้ไปมา:

- generated table ต้องอ่าน SEM จาก canonical statistic output
- manuscript checker compare formatted cells
- ไม่คำนวณหรือแก้ด้วยมือ

### 21.6 Test–retest .684 กับ .585

- .684 คือ correlation เมื่อ pool repeated pairs ทั้งหมด
- .585 คือ mean within user-domain correlation แล้ว cluster participants

สองค่าตอบคนละ estimand:

- pooled pairs วัด dataset-level association และให้น้ำหนัก units ที่มี pairs มากกว่า
- within-unit อยู่บน scale ใกล้ model evaluation ที่คำนวณ correlation ต่อ user-domain

อย่าใช้ .684 เป็น direct ceiling ของ Table 1 โดยไม่อธิบาย aggregation mismatch

### 21.7 Sparse models และ undefined correlations

Lasso/Elastic อาจ shrink coefficient จน prediction คงที่ในบาง unit ทำให้ correlation undefined

นี่ไม่ใช่ค่า 0 และไม่จำเป็นต้อง “ซ่อม” model เพื่อบังคับให้ได้เลข:

- เก็บ NaN
- รายงานจำนวน unit
- treat table as descriptive หากไม่ได้วาง inferential protocol
- อธิบายว่าเกิดจาก constant prediction ใน sparse fit

### 21.8 Comments อาจทำร้ายงาน

Comment เก่าที่ไม่ตรง code ปัจจุบันอันตรายกว่าการไม่มี comment

ก่อน release:

- grep TODO, FIXME, old, missing, leak, author names, absolute paths
- ตรวจ docstrings ของ core equations
- ลบ historical/defensive commentary
- รักษา comments ที่อธิบาย invariants

---

## 22. ขั้นตอนมาตรฐานสำหรับโปรเจกต์ใหม่

### Phase A: Scientific design

1. เขียนปัญหาในหนึ่งประโยค
2. เขียน tension
3. แตกเป็น RQs
4. ระบุ estimand ต่อ RQ
5. วาด executed computation graph
6. กำหนด unit และ independence
7. ออกแบบ matched controls
8. เขียน allowed claims ก่อนเห็นผล

Deliverable:

- research_design.md
- symbol table
- RQ matrix
- planned family map

### Phase B: Data protocol

1. ทำ dataset inventory
2. ระบุ duplicates/repeats/missing
3. กำหนด inclusion/exclusion
4. สร้าง split deterministic
5. ตรวจ user/image leakage
6. freeze split files
7. document scale/schema/license

Deliverable:

- DATA.md
- split files
- verification report

### Phase C: Modeling

1. implement simplest baseline
2. implement proposed representation
3. add matched controls
4. centralize preprocessing
5. centralize selection
6. log candidates
7. test on tiny slice
8. verify equations numerically

Numerical equation check:

- reconstruct prediction from saved terms
- compare max absolute error with model.predict
- run for random units/folds/domains

### Phase D: Evaluation

1. write unit-level rows
2. include config/seed/source
3. validate counts/keys
4. average seeds within unit
5. compute uncertainty at independent-unit level
6. run paired tests
7. apply pre-specified correction
8. store full-precision statistics

### Phase E: Diagnostics

1. owner specificity
2. input specificity
3. component sensitivity
4. agreement with observed data
5. peer/shared-direction null
6. failure cases
7. negative controls

แต่ละ diagnostic ต้องมี:

- exact intervention
- held-fixed variables
- estimand
- allowed claim
- alternative explanation

### Phase F: Paper

1. build result matrix
2. draft figure-first narrative
3. write Results จาก generated values
4. write Discussion หลัง Results freeze
5. write Introduction หลัง contribution ชัด
6. write Abstract สุดท้าย
7. run claim audit
8. render and visually inspect

### Phase G: Release

1. create anonymous clean repo
2. copy only required code/data
3. remove personal metadata
4. lock environment
5. run clean-room reproduce
6. diff all generated artifacts
7. archive tracked files
8. inspect archive contents
9. compute final checksum
10. freeze commit; user controls remote push

---

## 23. Templates สำหรับวางแผน experiment

### 23.1 Experiment card

    Name:
    Research question:
    Hypothesis:
    Estimand:
    Intervention:
    Reference:
    Held fixed:
    Unit:
    Pairing key:
    Sample size:
    Independence/clustering:
    Seeds and aggregation:
    Statistical test:
    Multiple-comparison family:
    Primary metric:
    Secondary metric:
    Expected output schema:
    Allowed positive claim:
    Allowed null claim:
    Known alternative explanations:
    Runtime:
    Freeze version:

### 23.2 Model component card

    Component:
    Input name and shape:
    Output name and shape:
    Training population:
    Validation population:
    Test population:
    Objective:
    Preprocessing:
    Hyperparameter grid:
    Selection metric:
    Random seed:
    Saved parameters:
    Reconstruction test:

### 23.3 Table/figure card

    Artifact:
    Source CSV:
    Generation function:
    Row/filter keys:
    Mean estimator:
    Uncertainty:
    Contrast:
    Corrected p-value source:
    Rounding:
    Caption unit statement:
    Manuscript references:

---

## 24. Definition of a defensible claim

ก่อนเก็บประโยคสำคัญไว้ใน paper ให้ตอบได้ครบ:

1. ตัวแปรไหนถูกเปลี่ยน
2. ตัวแปรไหนถูกตรึง
3. หน่วยที่เปรียบเทียบคืออะไร
4. ความไม่แน่นอนคำนวณอย่างไร
5. correction family คืออะไร
6. source rows อยู่ไฟล์ไหน
7. result regenerate ด้วย command ใด
8. alternative explanation สำคัญคืออะไร
9. wording จำกัด scope แล้วหรือไม่
10. abstract/conclusion พูดแรงกว่าผลหรือไม่

ถ้าตอบไม่ได้ ประโยคนั้นยังไม่พร้อมเป็น claim

---

## 25. สิ่งที่ทำให้งานนี้แข็งแรงจริง

จุดแข็งไม่ได้อยู่ที่ model complexity แต่อยู่ที่โครง argument:

- คำถาม interpretability ชัดและมี human-centered motivation
- ใช้ named concepts ที่มีฐานจาก aesthetic psychology
- แยก cost ของ bottleneck ด้วย Population, Direct, PCA และ Random
- ใช้ low-dimensional linear personal head ทำให้ intervention ตรวจได้
- มีทั้ง owner-, weight-, image- และ data-alignment diagnostics
- ประเมินหลาย backbones และ rating budgets
- ใช้ participant-disjoint protocol และ validation-user selection
- แก้ seed pseudo-replication และ cluster dependence
- นิยาม Holm families โปร่งใส
- release unit-level audit data และ downstream reproduction
- ผลลบช่วยชี้ Stage 1 perception bottleneck และ scope ของ prior

สิ่งเหล่านี้มีค่ากว่าเพิ่มโมเดลใหม่โดยไม่มี RQ รองรับ

---

## 26. สิ่งที่ยังเป็นขอบเขต ไม่ใช่ความล้มเหลว

- ใช้ dataset หลักหนึ่งชุด
- concept vocabulary มีเพียง 7 emotions
- Stage 1 ทำนาย consensus emotion ไม่ใช่ personal perception
- Anchor C มี raw-feature population branch จึงไม่ใช่ full concept bottleneck
- one deterministic support split ไม่วัด support-resampling variance
- GT reference optimistic และ selection protocol ต่าง
- qualitative explanation ยังไม่ได้ผ่าน human evaluation
- formulas เป็น association-based ไม่ใช่ causal psychological model
- generalization ข้าม culture/dataset ยังไม่ทราบ

การพูดขอบเขตเหล่านี้อย่างแม่นไม่ทำให้งานอ่อน แต่ช่วยกำหนดสิ่งที่งานตอบได้จริงและเปิด future work ที่มีเหตุผล

---

## 27. Future research directions ที่ต่อจากหลักฐาน

### 27.1 Improve Stage 1 emotion perception

เพราะ GT emotion head สูงกว่า predicted emotion pipeline มาก ทิศทางที่ตรงหลักฐานคือ:

- aesthetic-domain CLIP adaptation
- prompt/probe tuning
- parameter-efficient visual adaptation
- multi-task learning emotion + aesthetic attributes
- contrastive objectives ที่แยกอารมณ์ใกล้กัน
- uncertainty-aware mediator

ต้องทดสอบว่า gain มาจาก emotion accuracy จริง ไม่ใช่เพียง backbone ใหญ่ขึ้น

### 27.2 Combine response-side and image-side concepts

Emotions อธิบาย response ส่วน objective attributes อธิบาย stimulus

ออกแบบ factorial control:

- attributes only
- emotions only
- attributes + emotions
- same-width PCA
- Direct upper reference

ถามทั้ง accuracy และ whether each concept family explains residual variation beyond the other

### 27.3 Personalize perception and appraisal separately

ปัจจุบัน:

- shared h(x)
- personal delta in appraisal

อนาคต:

- h_u(x) = h_pop(x) + small personal perception correction
- appraisal remains linear/readable

ต้องออกแบบข้อมูลที่แยก:

- feeling differently about the same image
- weighting the same feeling differently in judgment

### 27.4 Human editing study

ถ้าจะอ้าง editable model ต้องทดสอบ:

- user understands formula
- user identifies incorrect weight
- editing weight changes desired images predictably
- accuracy/calibration after edit
- comparison with black-box feedback interface

### 27.5 Cross-dataset validation

อย่าฝืน dataset ที่ concepts ไม่สมดุลหรือกลายเป็น object-category proxy

ก่อนใช้ dataset ใหม่:

- label coverage
- class/prevalence balance
- annotators per image
- personal aesthetic targets
- repeated users across images
- domain/category confounding
- license
- whether aggregation destroys personalization signal

Binary emotion labels averaged across annotatorsให้ค่า prevalence/probability-like population response ไม่ใช่ intensity rating ของคนหนึ่งโดยตรง ต้องระบุ estimand ใหม่

---

## 28. Quick final checklist

### Before running

- [ ] RQ และ estimand เขียนแล้ว
- [ ] equation และ tensor shapes เขียนแล้ว
- [ ] control เปลี่ยนทีละ factor
- [ ] split/leakage tests ผ่าน
- [ ] validation protocol freeze
- [ ] statistical unit และ family map freeze

### After running

- [ ] expected unit count
- [ ] unique keys
- [ ] seed count
- [ ] NaN/constant prediction report
- [ ] source/config saved
- [ ] selection log saved
- [ ] equation reconstruction passed

### Before paper

- [ ] seed averaged before inference
- [ ] Average participant-clustered
- [ ] Holm corrected
- [ ] exact p-value inequalities
- [ ] no equivalence from null test
- [ ] table/figure generated
- [ ] all text numbers audited
- [ ] claim scope matches computation

### Before release

- [ ] clean checkout runs
- [ ] frozen outputs untouched
- [ ] no personal metadata
- [ ] no licensed data accidentally included
- [ ] archive contents inspected
- [ ] PDF visually inspected
- [ ] code, supplementary and main paper use same frozen result version

---

## 29. ประโยคเดียวที่ควรจำเมื่อทำงานต่อ

> อย่าถามเพียงว่า “เลขดีไหม”; ให้ถามว่าเลขนี้ประมาณสิ่งใด มาจากหน่วยใด ผ่าน intervention อะไร เทียบกับ control ใด มี dependence แบบไหน และอนุญาตให้เราเขียนได้แรงเพียงใด

เมื่อทุกคำตอบเชื่อมกัน งานจะทั้งแข็ง อ่านง่าย และป้องกันตัวเองได้โดยไม่ต้องเขียนแบบ defensive
