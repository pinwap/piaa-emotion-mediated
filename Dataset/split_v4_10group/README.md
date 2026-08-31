# split_v4_10group — leakage-free 5-fold split

สร้างด้วย `src/data/build_group_split.py` (deterministic — rerun ได้ผลเดิม)
แก้ปัญหา protocol 3 ข้อพร้อมกัน โดยใช้โครงสร้าง 10-group ของ XPASS (คอลัมน์ `set`)

## Design

XPASS มี **10 groups** (`set` 0–9) ที่ **user และ image ไม่ปนข้ามกลุ่ม**
(verified: user disjoint 100%, image disjoint 100%, ไม่มี user×image pair ข้าม group)

แต่ละ fold: **test = 2 groups, val = 1 group, train = 7 groups** — rotate test แบบไม่ทับ:

| fold | test groups | val | train groups | #test users | #val users | #train users | #GIAA images |
|---|---|---|---|---|---|---|---|
| 0 | {0,1} | 2 | {3,4,5,6,7,8,9} | 26 | 6  | 97 | 4,561 |
| 1 | {2,3} | 4 | {0,1,5,6,7,8,9} | 22 | 12 | 95 | 4,564 |
| 2 | {4,5} | 6 | {0,1,2,3,7,8,9} | 23 | 14 | 92 | 4,563 |
| 3 | {6,7} | 8 | {0,1,2,3,4,5,9} | 30 | 16 | 83 | 4,561 |
| 4 | {8,9} | 0 | {1,2,3,4,5,6,7} | 28 | 17 | 84 | 4,582 |

ทุก group เป็น test **ครบ 1 ครั้ง** (129 users ครอบคลุมพอดี)

## แก้ปัญหา protocol ครบ 3 ข้อ (verified — assertion ในสคริปต์ผ่านหมด)

1. **hyperparameter ไม่ leak** — เลือกจาก **val users** (แยกจาก test users สนิท) แล้ว fix ใช้ทุก test user
2. **image ไม่ leak** — GIAA เทรนเฉพาะ train-group images; `giaa_train_images ∩ test/val images = 0` ทุก fold
3. **unseen users + unseen images พร้อมกัน** — test user ทั้งคน + ภาพเขาไม่เคยเข้า train เลย + มี fold variance (mean±sd)

## ไฟล์ต่อ fold (`fold{0-4}/`)
- `train_users.txt` / `val_users.txt` / `test_users.txt` — user_id ต่อบรรทัด
- `giaa_train_images.txt` — sample_id ของ train groups (สำหรับเทรน GIAA / shared emotion model)
- `meta.json` — test/val/train groups + จำนวน

## วิธีเอาไปใช้ (สำหรับ re-run — ยังไม่ทำ)
1. เทรน GIAA / shared emotion model บน `giaa_train_images` เท่านั้น
2. เลือก hyperparameter (alpha, adaptation steps) บน **val users** → fix
3. per test user: แบ่ง ratings เป็น adaptation (k=10/25/50/100) + eval (ที่เหลือ) — ทั้งคู่เป็นภาพ test-group
4. รายงาน mean ± sd ข้าม 5 folds

**หมายเหตุ:** ต่างจาก split เดิม (`v3_fold`) ที่ val มาจาก test users เอง + GIAA images ทับ PIAA images 90%
→ split นี้ตัดปัญหาทั้งสอง แต่ pipeline ต้องแก้ + re-run ทุก experiment (ตัวเลขจะเปลี่ยน)
