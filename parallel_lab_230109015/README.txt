# Quick start — Student ID 230109015

1. Extract this ZIP file.
2. Open the extracted `parallel_lab_230109015` folder in VS Code.
3. Open the integrated terminal in that folder.
4. Run the phases in this order:

```powershell
python task0_cal.py
python task1_falsesharing.py
python task2_scaling.py
python task3a_roofline.py
python task3b_gemm.py
python task4_scheduling.py
```

When a script asks for the physical core count, enter the **physical core** number from Windows Task Manager or this PowerShell command:

```powershell
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, L2CacheSize, L3CacheSize
```

For more hardware information, open Task Manager → Performance → CPU. Windows may not expose L1 cache size through the command above; check the CPU manufacturer's specifications or Windows system information and record the source you used.

Important:
- Run one phase at a time and let it finish.
- These are intentionally heavy benchmarks. Phase 1, Phase 2, and the GEMM tests can take a long time. Keep the laptop plugged in and close other heavy applications.
- Do not change the student ID constant in the scripts.
- Copy actual terminal output and actual `VERIFY_HASH` lines into `lab_report_230109015.md`.
- The code produces measurements; it cannot truthfully fill in your CPU model, cache sizes, or measured results in advance.
- The supplied scripts implement the requested experiments as a practical starting point. Compare the exact experiment wording in your handout with the code and your instructor's expectations before submission.
