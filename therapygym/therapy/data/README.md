# Data directory

TherapyGym's patient profiles and third-party character cards are not included in
the code repository. Obtain each dataset under its applicable license and place
the files used by the preparation scripts here:

- `patient_psi_cm_dataset.json`: a JSON array of patient-profile objects. Each
  object must contain a unique string-valued `id` field and the profile fields
  consumed by the patient simulator.
- `esc_character_cards_en.json`: the licensed English ESC character cards as a
  JSON array.
- `esc_character_cards_zh.json`: the licensed Chinese ESC character cards as a
  JSON array.

Do not commit identifiable patient information, API credentials, or data whose
license does not permit redistribution.
