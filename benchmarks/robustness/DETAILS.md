# Latest Qwen3-ASR Audio Robustness Details

- Version: `0.3.0-snapshot`
- Model: `Qwen/Qwen3-ASR-0.6B-hf`
- GPU: `NVIDIA GeForce RTX 5070 Ti (index 0, 16303 MiB, driver 610.88, GPU-1e924b6b-3d80-bedf-061e-ea4fd7ed892c)`
- Run time: `03.10.2026 22:20:46`
- Passed: `22/44`
- Auto non-speech false positives: `0/10`
- Forced-language non-speech false positives: `18/20`
- Speech passed: `10/14`

| Fixture | Category | Request language | Result | Score | Expected | Actual |
| --- | --- | --- | --- | ---: | --- | --- |
| silence_1s | silence | Auto | PASS | 100.00% | (empty) | (empty) |
| silence_1s | silence | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| silence_1s | silence | English | FAIL | 0.00% | (empty) | Okay. |
| silence_5s | silence | Auto | PASS | 100.00% | (empty) | (empty) |
| silence_5s | silence | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| silence_5s | silence | English | FAIL | 0.00% | (empty) | The. |
| silence_15s | silence | Auto | PASS | 100.00% | (empty) | (empty) |
| silence_15s | silence | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| silence_15s | silence | English | FAIL | 0.00% | (empty) | The. |
| white_noise_low | noise | Auto | PASS | 100.00% | (empty) | (empty) |
| white_noise_low | noise | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| white_noise_low | noise | English | FAIL | 0.00% | (empty) | The. |
| white_noise_medium | noise | Auto | PASS | 100.00% | (empty) | (empty) |
| white_noise_medium | noise | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| white_noise_medium | noise | English | PASS | 100.00% | (empty) | (empty) |
| white_noise_loud | noise | Auto | PASS | 100.00% | (empty) | (empty) |
| white_noise_loud | noise | Chinese | FAIL | 0.00% | (empty) | 啊。 |
| white_noise_loud | noise | English | PASS | 100.00% | (empty) | (empty) |
| pink_noise_medium | noise | Auto | PASS | 100.00% | (empty) | (empty) |
| pink_noise_medium | noise | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| pink_noise_medium | noise | English | FAIL | 0.00% | (empty) | The. |
| brown_noise_medium | noise | Auto | PASS | 100.00% | (empty) | (empty) |
| brown_noise_medium | noise | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| brown_noise_medium | noise | English | FAIL | 0.00% | (empty) | The. |
| voiceband_noise | noise | Auto | PASS | 100.00% | (empty) | (empty) |
| voiceband_noise | noise | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| voiceband_noise | noise | English | FAIL | 0.00% | (empty) | The. |
| mains_hum_60hz | noise | Auto | PASS | 100.00% | (empty) | (empty) |
| mains_hum_60hz | noise | Chinese | FAIL | 0.00% | (empty) | 嗯。 |
| mains_hum_60hz | noise | English | FAIL | 0.00% | (empty) | The. |
| english_echo_moderate | echo | Auto | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| english_echo_moderate | echo | English | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| english_echo_severe | echo | Auto | FAIL | 78.26% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make him come up to me. He said, "I'll pounce on you on the list." Very British, very very very serious. |
| english_echo_severe | echo | English | FAIL | 78.26% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make him come up to me. He said, "I'll pounce on you on the list." Very British, very very very serious. |
| english_weak_20db | weak_speech | Auto | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| english_weak_20db | weak_speech | English | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| english_weak_30db | weak_speech | Auto | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| english_weak_30db | weak_speech | English | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| english_weak_40db | weak_speech | Auto | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| english_weak_40db | weak_speech | English | PASS | 100.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | I tried to make a cup of tea, but the kettle said, "I'll put you on the list." Very British, very serious. |
| chinese_weak_40db | weak_speech | Auto | PASS | 97.67% | 今天的奶茶很懂事，三分糖，却给了我十分的快乐。 | 今天的奶茶很懂事，三分糖却给了我十分的快乐。 |
| chinese_weak_40db | weak_speech | Chinese | PASS | 97.67% | 今天的奶茶很懂事，三分糖，却给了我十分的快乐。 | 今天的奶茶很懂事，三分糖却给了我十分的快乐。 |
| english_weak_noisy | weak_speech | Auto | FAIL | 0.00% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | (empty) |
| english_weak_noisy | weak_speech | English | FAIL | 5.88% | I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' Very British, very serious. | The. |
