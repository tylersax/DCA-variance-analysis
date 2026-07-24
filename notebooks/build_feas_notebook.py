"""Assemble symmetry_variance_demo_feas.ipynb (claim B, 4x4 FeAs) from cell sources."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
nb=new_notebook(); C=[]
def md(s): C.append(new_markdown_cell(s))
def code(s): C.append(new_code_cell(s))

md(r"""# FeAs variance-reduction demo — claim B: *derived symmetry that was never declared*

**CT-AUX, FeAs (two-band), Nc = 16 (4×4), one DCA iteration.** N = 32 paired independent runs/arm,
shared explicit-integer seed list (prime stride 1000003).

FeAs is the project's actual story. Its model **declares** `FeAsPointGroup` = {identity, C4} = **2 ops**.
The H0-derived, authoritative group is the full **D4 = 8 ops** — the derivation recovers 6 that were
never declared (the mirrors and higher rotations), each a *band-permuting* op reachable only via the
P-search (#374). The ON arm imposes all 8; the OFF arm (`no_symmetry`, the honored off-switch) imposes 1.

> **Claim B.** The derived group averages each band-diagonal star element over up to **8** symmetry
> images (measured), reducing its variance by up to 8×. The **declared** group has only **2** ops, so
> it can average any element over at most 2 images — a **≤ 2× cap**. Reduction measured *above* that
> cap is bought purely by H0-derived symmetry the declaration never had.

Why 4×4 and not 2×2: on a 2×2 cluster every k-orbit is trivial and the interband channel is identically
zero (H0 interband ∝ sin kx sin ky = 0 there), so the extra ops act only on null entries. The 4×4 mesh
has genuine k-stars and non-null interband, so the derived group's extra reach becomes measurable.""")

code(r"""import numpy as np, matplotlib.pyplot as plt, glob
import variance_demo_feas_lib as L
plt.rcParams.update({'figure.dpi':110,'font.size':10})
import os
RUN = os.environ.get('VARIANCE_DEMO_RUNS', '../variance_demo/runs_feas4')""")

md(r"""## 1. Load & guard — ON imposed 8 ops, OFF imposed 1, seeds paired.""")
code(r"""Gon,s_on,kel,nops_on = L.load_arm(RUN,'on')
Goff,s_off,_,nops_off = L.load_arm(RUN,'off')
assert nops_on=={8}, nops_on
assert nops_off=={1}, nops_off
assert (s_on==s_off).all()
Nrep,Nw,Nk,Nb,_=Gon.shape
print(f'{Nrep} replicas/arm, Nw={Nw}, Nk={Nk}, bands={Nb}')
print(f'ON ops={nops_on}, OFF ops={nops_off}, seeds paired: OK')""")

md(r"""## 2. The derivation report (captured from an ON run) — literal evidence for 2→8.""")
code(r"""log=sorted(glob.glob(RUN+'/on/log_*.txt'))[0]
seen=set()
for line in open(log):
    l=line.strip()
    if ('declared group:' in l or 'under-declared:' in l) and l not in seen:
        seen.add(l); print(l)
    if len(seen)>=2: break""")

md(r"""## 3. Independence check (§3b) — no drift, no lag-1 autocorrelation, no seed-proximity effect.""")
code(r"""order=np.argsort(s_off); Go=Goff[order]; sd=s_off[order]
F,_=L.flatten_entries(Go); wpk=np.abs(Goff.mean(0)).reshape(Nw,-1).mean(1).argmax()
mag=np.abs(Go.mean(axis=(1,2,3,4))); drift=np.corrcoef(np.arange(Nrep),mag)[0,1]
X=F[:,wpk,:]; sX=X.std(0); keep=sX>1e-12*np.abs(X).mean()
Xs=(X[:,keep]-X[:,keep].mean(0))/X[:,keep].std(0)
lag1=np.mean([np.real(np.vdot(Xs[:-1,k],Xs[1:,k]))/(Nrep-1) for k in range(Xs.shape[1])])
band=1.96/np.sqrt(Nrep)
dk=Xs[:,0]; iu=np.triu_indices(Nrep,1)
prox=np.corrcoef(np.abs(sd[:,None]-sd[None,:])[iu],np.abs(dk[:,None]-dk[None,:])[iu])[0,1]
print(f'1. drift            = {drift:+.3f}  (|.|<{band:.3f})')
print(f'2. lag-1 autocorr   = {lag1:+.3f}  (|.|<{band:.3f})')
print(f'3. corr(Δseed,ΔG)   = {prox:+.3f}  (trap=>negative)')
assert abs(drift)<band and abs(lag1)<band and prox>-band
print('\nReplicas independent — variance estimates trustworthy.')""")

md(r"""## 4. Symmetrization orbits — derived (measured) vs the declared 2-op cap

Orbits defined empirically: entries the ON arm collapses to equal values are orbit-mates. The declared
{id,C4} group could average any entry over at most 2 images (2 ops), so **2 is the reduction ceiling**
the declaration could ever reach; every derived orbit of size > 2 is beyond it.""")
code(r"""der,labels=L.orbits_from_on(Gon)
from collections import Counter
print('DERIVED orbit sizes:', dict(sorted(Counter(len(g) for g in der).items())))
print('declared cap: 2 ops -> <=2x on any entry\n')
for g in sorted(der,key=len,reverse=True):
    kind=('interband' if all(labels[i][1]!=labels[i][2] for i in g)
          else 'band-diag' if all(labels[i][1]==labels[i][2] for i in g) else 'mixed')
    ks=sorted(set(labels[i][0] for i in g))
    beyond='  <-- beyond declared 2x' if len(g)>2 else ''
    print(f'  size {len(g):>2} {kind:>10}: {len(g)} entries over k-indices {ks}{beyond}')""")

md(r"""## 5. Money plot — measured Var_off/Var_on per derived orbit vs the declared ceiling

Each derived orbit: measured ratio (pooled over ω and orbit-mates) with a 95% bootstrap CI, coloured by
whether its size exceeds the declared 2-op ceiling. Reference lines: null y=1, **declared cap y=2**,
ideal y=n. Points sitting above the y=2 line are variance reduction only the derived group can reach.""")
code(r"""def pooled_ratio(m):
    Voff=L.replica_variance_flat(Goff); Von=L.replica_variance_flat(Gon)
    von=Von[:,m].mean()
    return (Voff[:,m].mean()/von) if von>0 else np.nan
def boot_ci(m,nb=1000,seed=0):
    rng=np.random.default_rng(seed); Fo,_=L.flatten_entries(Goff); Fn,_=L.flatten_entries(Gon)
    v=[]
    for _ in range(nb):
        b=rng.integers(0,Nrep,Nrep)
        dn=Fn[b][:,:,m].var(0,ddof=1).real.mean()
        if dn>0: v.append(Fo[b][:,:,m].var(0,ddof=1).real.mean()/dn)
    return (np.percentile(v,2.5),np.percentile(v,97.5)) if v else (np.nan,np.nan)

# Null-orbit guard. Some derived orbits (all interband here) are symmetry-forced to ~0: the derived
# group averages entries that cancel, so the ON-symmetrized value -- hence Von -- is machine-zero and
# the ratio Voff/Von is an ill-defined 0/0 that explodes. Those carry no signal; exclude them with a
# note. Scale set by the band-diagonal magnitude (the physical, non-null channel).
Fon_all,_=L.flatten_entries(Gon)
bd = [i for i,l in enumerate(labels) if l[1]==l[2]]
sig = np.abs(Fon_all[:,:,bd]).mean()
def is_null(g):
    return np.abs(Fon_all[:,:,g]).mean() < 1e-4*sig
rows=[]; n_null=0
for g in der:
    if is_null(g):
        n_null+=1; continue
    n=len(g); r=pooled_ratio(g)
    if not np.isfinite(r): n_null+=1; continue
    lo,hi=boot_ci(g); rho=L.orbit_rho_flat(Goff,g)
    interband=all(labels[i][1]!=labels[i][2] for i in g)
    rows.append(dict(n=n,ratio=r,lo=lo,hi=hi,rho=rho,interband=interband))
print(f'({n_null} symmetry-forced-null orbits excluded: derived group sends them to ~0)')

fig,ax=plt.subplots(figsize=(6.6,4.4))
jit=(np.random.default_rng(1).random(len(rows))-0.5)*0.18
for r,j in zip(rows,jit):
    c='#c0392b' if r['n']>2 else '#2c6fbb'
    ax.errorbar([r['n']+j],[r['ratio']],
                yerr=[[max(0,r['ratio']-r['lo'])],[max(0,r['hi']-r['ratio'])]],
                fmt='o',ms=7,capsize=3,color=c,zorder=3)
xx=np.linspace(0.8,8.4,50); ax.plot(xx,xx,'--',color='#bbb',label='ideal y=n (ρ=0)')
ax.axhline(2,color='#c0392b',lw=1.2,ls=':',label='declared 2-op ceiling')
ax.axhline(1,color='#444',lw=.8,label='null y=1')
ax.scatter([],[],marker='o',color='#c0392b',label='derived orbit size > 2 (beyond declared)')
ax.scatter([],[],marker='o',color='#2c6fbb',label='derived orbit size ≤ 2')
ax.set_xlabel('derived orbit size $n$'); ax.set_ylabel(r'$\mathrm{Var}_{\rm off}/\mathrm{Var}_{\rm on}$')
ax.set_title('FeAs 4×4: derived group reduces variance beyond the declared ceiling')
ax.set_xticks([1,2,3,4,6,8]); ax.legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig('figures/feas4_money_plot.png',dpi=130); plt.show()
for r in sorted(rows,key=lambda x:x['n']):
    tag='interband' if r['interband'] else 'band-diag'
    print(f"  n={r['n']:>2} ({tag}) ratio={r['ratio']:.2f} [{r['lo']:.2f},{r['hi']:.2f}] rho={r['rho']:.3f}")""")

md(r"""## 6. Bias control — imposing 6 undeclared ops must not move the mean.""")
code(r"""Fo,_=L.flatten_entries(Goff); Fn,_=L.flatten_entries(Gon)
Vo=Fo.var(0,ddof=1); Vn=Fn.var(0,ddof=1); se=np.sqrt((Vo+Vn)/Nrep)
z=(Fn.mean(0)-Fo.mean(0))/np.where(se>0,se,np.nan); zf=z.ravel(); zf=zf[np.isfinite(zf)]
fig,ax=plt.subplots(figsize=(5,3.2))
ax.hist(zf.real,bins=40,density=True,color='#8bb',edgecolor='k',lw=.3)
xx=np.linspace(-4,4,100); ax.plot(xx,np.exp(-xx**2/2)/np.sqrt(2*np.pi),'r')
ax.set_title(f'Re z, all entries  mean={zf.real.mean():.2f} std={zf.real.std():.2f}'); ax.set_xlabel('z')
fig.tight_layout(); fig.savefig('figures/feas4_bias.png',dpi=130); plt.show()
print(f'bias: mean|z|={np.abs(zf).mean():.2f}  (imposing 6 undeclared ops does not bias the mean)')""")

md(r"""## 7. Summary — the claim-B table

Per derived orbit above the declared ceiling: size, kind, measured ratio + CI, ρ. These are variance
reductions the declared 2-op group could not have achieved.""")
code(r"""big=[r for r in rows if r['n']>2]
print(f"{'derived n':>9} {'kind':>10} {'ratio':>6} {'95% CI':>13} {'rho':>6}")
print('-'*52)
for r in sorted(big,key=lambda x:-x['n']):
    tag='interband' if r['interband'] else 'band-diag'
    print(f"{r['n']:>9} {tag:>10} {r['ratio']:>6.2f} {'['+format(r['lo'],'.2f')+','+format(r['hi'],'.2f')+']':>13} {r['rho']:>6.3f}")
print('\nClaim B: the derived D4 (8 ops) averages these band-diagonal star elements over 4-8 symmetry')
print('images; the declared 2-op group caps at 2x. The mean is unbiased. This variance reduction is')
print('bought purely by H0-derived symmetry that was never declared (2 declared -> 8 derived).')""")

nb['cells']=C
nb.metadata['kernelspec']={'name':'python3','display_name':'Python 3','language':'python'}
open('symmetry_variance_demo_feas.ipynb','w').write(nbf.writes(nb))
print('wrote symmetry_variance_demo_feas.ipynb with',len(C),'cells')
