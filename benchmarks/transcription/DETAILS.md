# Qwen3-ASR Transcription Benchmark Details

Each benchmark run appends a section with representative best and worst examples.

This benchmark uses the random-only testbench corpus: 30 Qwen3-ASR-supported languages, 10 samples per language, 300 cases total. Each official run starts with 10 mandatory prewarm inferences; prewarm results and time are discarded. Official timing records the measured benchmark run only. Expected and actual examples bold the differing spans so regressions are easier to inspect.

## 15.05.2026 18:46:00 - Qwen/Qwen3-ASR-0.6B

- Version: `0.1-snapshot`
- Total score: `96.10%`
- Bonus: `0.40%`
- Total time: `39.532s`
- Cases: `300`

### Best Examples

#### chinese_random_04 - 102.50%

- Language: `Chinese`
- Expected: 人生就像火锅，别急着下结论，先看看谁被烫到了。
- Actual: 人生就像火锅，别急着下结论，先看看谁被烫到了。

#### chinese_random_05 - 102.50%

- Language: `Chinese`
- Expected: 我不是拖延症，我只是给灵感一点自由活动时间。
- Actual: 我不是拖延症，我只是给灵感一点自由活动时间。

#### chinese_random_06 - 102.50%

- Language: `Chinese`
- Expected: **电梯到了，门一开，尴尬也一起上来了。 [sigh] 尴尬还按了最近的楼层。**
- Actual: **电梯到了，门一开，尴尬也一起上来了。尴尬还按了最近的楼层。**

### Worst Examples

#### cantonese_random_02 - 71.25%

- Language: `Cantonese`
- Expected: 我**買**咗一杯**熱**奶茶，坐喺窗**邊聽**雨**聲**。
- Actual: 我**买**咗一杯**热**奶茶，坐喺窗**边听**雨**声**。

#### cantonese_random_06 - 71.25%

- Language: `Cantonese`
- Expected: **廚**房**傳**嚟**飯**香，大家都自然行近**張枱**。
- Actual: **厨**房**传**嚟**饭**香，大家都自然行近**张台**。

#### cantonese_random_01 - 73.09%

- Language: `Cantonese`
- Expected: 今日**個**公**園**好安**靜**，我慢慢行**過條**石路。
- Actual: 今日**个**公**园**好安**静**，我慢慢行**过条**石路。


## 15.05.2026 18:46:46 - Qwen/Qwen3-ASR-0.6B

- Version: `0.1-snapshot`
- Total score: `96.10%`
- Bonus: `0.40%`
- Total time: `39.815s`
- Cases: `300`

### Best Examples

#### chinese_random_04 - 102.50%

- Language: `Chinese`
- Expected: 人生就像火锅，别急着下结论，先看看谁被烫到了。
- Actual: 人生就像火锅，别急着下结论，先看看谁被烫到了。

#### chinese_random_05 - 102.50%

- Language: `Chinese`
- Expected: 我不是拖延症，我只是给灵感一点自由活动时间。
- Actual: 我不是拖延症，我只是给灵感一点自由活动时间。

#### chinese_random_06 - 102.50%

- Language: `Chinese`
- Expected: **电梯到了，门一开，尴尬也一起上来了。 [sigh] 尴尬还按了最近的楼层。**
- Actual: **电梯到了，门一开，尴尬也一起上来了。尴尬还按了最近的楼层。**

### Worst Examples

#### cantonese_random_02 - 71.25%

- Language: `Cantonese`
- Expected: 我**買**咗一杯**熱**奶茶，坐喺窗**邊聽**雨**聲**。
- Actual: 我**买**咗一杯**热**奶茶，坐喺窗**边听**雨**声**。

#### cantonese_random_06 - 71.25%

- Language: `Cantonese`
- Expected: **廚**房**傳**嚟**飯**香，大家都自然行近**張枱**。
- Actual: **厨**房**传**嚟**饭**香，大家都自然行近**张台**。

#### cantonese_random_01 - 73.09%

- Language: `Cantonese`
- Expected: 今日**個**公**園**好安**靜**，我慢慢行**過條**石路。
- Actual: 今日**个**公**园**好安**静**，我慢慢行**过条**石路。


## 15.05.2026 18:53:32 - Qwen/Qwen3-ASR-1.7B

- Version: `0.1-snapshot`
- Total score: `85.97%`
- Bonus: `0.59%`
- Total time: `106.042s`
- Cases: `300`

### Best Examples

#### chinese_random_04 - 102.50%

- Language: `Chinese`
- Expected: 人生就像火锅，别急着下结论，先看看谁被烫到了。
- Actual: 人生就像火锅，别急着下结论，先看看谁被烫到了。

#### chinese_random_05 - 102.50%

- Language: `Chinese`
- Expected: 我不是拖延症，我只是给灵感一点自由活动时间。
- Actual: 我不是拖延症，我只是给灵感一点自由活动时间。

#### chinese_random_06 - 102.50%

- Language: `Chinese`
- Expected: **电梯到了，门一开，尴尬也一起上来了。 [sigh] 尴尬还按了最近的楼层。**
- Actual: **电梯到了，门一开，尴尬也一起上来了。尴尬还按了最近的楼层。**

### Worst Examples

#### polish_random_07 - 4.91%

- Language: `Polish`
- Expected: **Posprzątałem biurko i zgubiłem cały system porządnego bałaganu.**
- Actual: **我不是拖延症，我只是给灵感一点自由活动时间。**

#### filipino_random_02 - 5.28%

- Language: `Filipino`
- Expected: **Naglagay ako ng kape sa mesa at binasa ang lumang liham.**
- Actual: **今日个公园好安静，我慢慢行过条石路。**

#### filipino_random_10 - 5.36%

- Language: `Filipino`
- Expected: **Sa gabi, nakinig ako sa radyo habang naghuhugas ng tasa.**
- Actual: **厨房传嚟饭香，大家都自然行近张台。**


## 15.05.2026 19:15:42 - Qwen/Qwen3-ASR-0.6B

- Version: `0.1-snapshot`
- Total score: `96.04%`
- Bonus: `0.39%`
- Total time: `41.942s`
- Cases: `300`

### Best Examples

#### chinese_random_04 - 102.50%

- Language: `Chinese`
- Expected: 人生就像火锅，别急着下结论，先看看谁被烫到了。
- Actual: 人生就像火锅，别急着下结论，先看看谁被烫到了。

#### chinese_random_05 - 102.50%

- Language: `Chinese`
- Expected: 我不是拖延症，我只是给灵感一点自由活动时间。
- Actual: 我不是拖延症，我只是给灵感一点自由活动时间。

#### chinese_random_06 - 102.50%

- Language: `Chinese`
- Expected: **电梯到了，门一开，尴尬也一起上来了。 [sigh] 尴尬还按了最近的楼层。**
- Actual: **电梯到了，门一开，尴尬也一起上来了。尴尬还按了最近的楼层。**

### Worst Examples

#### cantonese_random_02 - 71.25%

- Language: `Cantonese`
- Expected: 我**買**咗一杯**熱**奶茶，坐喺窗**邊聽**雨**聲**。
- Actual: 我**买**咗一杯**热**奶茶，坐喺窗**边听**雨**声**。

#### cantonese_random_06 - 71.25%

- Language: `Cantonese`
- Expected: **廚**房**傳**嚟**飯**香，大家都自然行近**張枱**。
- Actual: **厨**房**传**嚟**饭**香，大家都自然行近**张台**。

#### cantonese_random_01 - 73.09%

- Language: `Cantonese`
- Expected: 今日**個**公**園**好安**靜**，我慢慢行**過條**石路。
- Actual: 今日**个**公**园**好安**静**，我慢慢行**过条**石路。


## 15.05.2026 19:22:15 - Qwen/Qwen3-ASR-1.7B

- Version: `0.1-snapshot`
- Total score: `97.64%`
- Bonus: `0.40%`
- Total time: `84.751s`
- Cases: `300`

### Best Examples

#### chinese_random_04 - 102.50%

- Language: `Chinese`
- Expected: 人生就像火锅，别急着下结论，先看看谁被烫到了。
- Actual: 人生就像火锅，别急着下结论，先看看谁被烫到了。

#### chinese_random_05 - 102.50%

- Language: `Chinese`
- Expected: 我不是拖延症，我只是给灵感一点自由活动时间。
- Actual: 我不是拖延症，我只是给灵感一点自由活动时间。

#### chinese_random_06 - 102.50%

- Language: `Chinese`
- Expected: **电梯到了，门一开，尴尬也一起上来了。 [sigh] 尴尬还按了最近的楼层。**
- Actual: **电梯到了，门一开，尴尬也一起上来了。尴尬还按了最近的楼层。**

### Worst Examples

#### cantonese_random_02 - 71.25%

- Language: `Cantonese`
- Expected: 我**買**咗一杯**熱**奶茶，坐喺窗**邊聽**雨**聲**。
- Actual: 我**买**咗一杯**热**奶茶，坐喺窗**边听**雨**声**。

#### cantonese_random_06 - 71.25%

- Language: `Cantonese`
- Expected: **廚**房**傳**嚟**飯**香，大家都自然行近**張枱**。
- Actual: **厨**房**传**嚟**饭**香，大家都自然行近**张台**。

#### cantonese_random_10 - 71.25%

- Language: `Cantonese`
- Expected: 我**將**窗打**開**，俾清**風**同**陽**光一**齊**入屋。
- Actual: 我**将**窗打**开**，俾清**风**同**阳**光一**齐**入屋。
