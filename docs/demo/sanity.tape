# Records docs/demo/sanity.gif from a real run: `vhs docs/demo/sanity.tape` from the repo root.
Output docs/demo/sanity.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 620
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' RUNS=../.bench-data/adversarial-robustness-lab/runs && source .venv/bin/activate && clear` Enter
Sleep 2s
Show
Type `python -m adversarial_robustness_lab sanity -c $RUNS/wrn16_2_pgd.pt --quantize-bits 3 --n 200 --queries 300` Enter
Wait+Line@300s /lab\) \$/
Sleep 2s
Type `python -m adversarial_robustness_lab evaluate -c $RUNS/wrn16_2_pgd.pt --quantize-bits 3 --bpda --n 200` Enter
Wait+Line@180s /lab\) \$/
Sleep 4s
