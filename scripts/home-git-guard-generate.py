import subprocess,os,re,sys
home=os.path.expanduser("~"); ex=os.path.join(home,".git","info","exclude")
tracked=subprocess.run(["git","-C",home,"ls-files"],capture_output=True,text=True,check=True).stdout.split("\n")
tracked=[t for t in tracked if t]
BEGIN="# home-git-guard-begin (2026-10-16): the ONLY paths that may be visible to `git ls-files --others` in the home repo are the tracked files; generated, rerun the generator to refresh"
END="# home-git-guard-end"
# directories that must be traversed = ancestors of tracked files; everything else under them is excluded
dirs=set(); 
for t in tracked:
    parts=t.split("/")
    for i in range(1,len(parts)): dirs.add("/".join(parts[:i]))
rules=[BEGIN,"/*"]
children={}
for d in sorted(dirs|set(tracked)):
    parent="/".join(d.split("/")[:-1]); children.setdefault(parent,set()).add(d)
def emit(d):
    kids=sorted(children.get(d,[]))
    if not kids: return
    if d: rules.append("/%s/*"%d)
    for k in kids:
        rules.append("!/%s%s"%(k,"/" if k in dirs else ""))
        if k in dirs: emit(k)
emit("")
rules.append(END)
src=open(ex,encoding="utf-8").read()
pat=re.compile(re.escape(BEGIN)+r".*?"+re.escape(END)+r"\n?",re.S)
src=pat.sub("",src).rstrip("\n")+"\n"
# remove the earlier hand-written block (the /* ... lines after the 2026-10-15 comment) so the generated one is the only authority
old=re.search(r"(# 2026-10-15 \(Livestream Manager.*?\n)(/\*\n(?:!?/\.copilot[^\n]*\n)+)",src,re.S)
if old:
    src=src.replace(old.group(2),"")
    src=src.replace(old.group(1),old.group(1)+"# (the hand-written rules that followed were replaced by the generated block below; original in exclude.bak-20261015)\n")
src+="\n".join(rules)+"\n"
open(ex,"w",encoding="utf-8",newline="\n").write(src)
print("wrote %d rules for %d tracked files"%(len(rules),len(tracked)))
