# Auditoria do módulo de transporte: gases puros e misturas

**Data:** 2026-09-10 · **Versão auditada:** statthermopy 0.1.0 · **Código não foi alterado.**

---

## 0. Correção de premissa

O pedido parte de **23 gases**. O banco tem **30**, extraídos programaticamente:

```
AR, C2H2, C2H4, C2H6, C3H8, C6H6, CCL4, CFCL3, CH3CCL3, CH4, CL2, CO, CO2,
H2, H2O, H2S, HBR, HCL, HE, HI, I2, KR, N2, N2O, NE, NH3, NO, O2, SO2, XE
```

Eram 22 até 2026-09-09, quando 8 espécies foram acrescentadas (I2, HCl, HBr, HI, CCl4, CFCl3,
CH3CCl3, C6H6). O número 23 não corresponde a nenhum estado do repositório. **Toda esta auditoria
cobre as 30 espécies reais.**

---

## 1. CURRENT IMPLEMENTATION

### 1.1 Gás puro — `transport/transport.py`

Chapman–Enskog de primeira ordem com potencial Lennard-Jones 12-6, integrais de colisão de
Neufeld–Janzen–Aziz (1972):

| Propriedade | Fórmula | Insumos |
|---|---|---|
| μ | (5/16)·√(m k_B T/π) / (σ²Ω^(2,2)*(T*)) | σ_i, ε_i, M_i da espécie |
| k | μ·c_v·(9γ−5)/4 (Eucken) | μ_i + C_v,γ da função de partição |
| D_ii | (3/16)(k_BT/P)/(σ²Ω^(1,1)*)·√(2k_BT/πm_ii) | σ_i, ε_i, M_i |
| D_ij | idem com Lorentz–Berthelot | σ_ij=(σ_i+σ_j)/2, ε_ij=√(ε_iε_j), m_ij |
| ρ | PM/(RT) | M_i |
| ν, α | μ/ρ, k/(ρc_p) | derivadas |
| Pr | 4γ/(9γ−5) (forma fechada de Eucken) | γ_i |
| Sc | (5/6)·Ω^(1,1)*/Ω^(2,2)* | T*_i |
| Le | Sc/Pr | — |
| Z, a, β, κ_T, μ_JT | 1, √(γRT/M), 1/T, 1/P, 0 | gás ideal exato |

**Nenhum dado tabelado de transporte é consumido.** C_v, C_p e γ vêm da função de partição, de
modo que a dependência quântica de temperatura propaga para μ, k, Pr.

### 1.2 Mistura — `transport/air/mixture_transport.py`

| Propriedade | Regra | Genérica para N? |
|---|---|---|
| μ_mix | **Wilke (1950)**: Σ_i x_iμ_i/(Σ_j x_jφ_ij) | ✅ laços `range(n)` |
| k_mix | **Mason–Saxena (1958)**, forma de Wilke com k_i | ✅ |
| D_i,mix | **Blanc**: (1−x_i)/Σ_{j≠i}(x_j/D_ij) | ✅ matriz n×n |
| M_mix | Σ x_i M_i | ✅ |
| ρ_mix | P·M_mix/(RT) | ✅ |
| Cp_mix, Cv_mix, γ | `IdealGasMixture.compute` | ✅ |
| R_mix | R_u/M_mix | ✅ |
| ν, α | μ_mix/ρ, k_mix/(ρ c_p) | ✅ |
| Pr, Sc, Le | μc_p/k, ν/D_eff, α/D_eff | ✅ |

φ_ij = [1+(v_i/v_j)^½(M_j/M_i)^¼]²/√(8(1+M_i/M_j)), com v = μ (Wilke) ou k (Mason–Saxena).

### 1.3 Ar — `transport/air/air_transport.py`

Fachada fina. `AirTransport.dry()` monta a mistura de 4 componentes; `.humid()` resolve a
composição via `HumidAir.state()` e alimenta o mesmo calculador genérico. **Nenhuma física é
reimplementada aqui.**

---

## 2. GAS INVENTORY

Todas as 30 espécies possuem **σ e ε/k próprios e distintos**. Nenhuma duplicata em nenhum dos
dois parâmetros.

| Species | Formula | M [g/mol] | Thermo | Transport | σ [Å] | ε/k [K] | Source |
|---|---|---:|---|---|---:|---:|---|
| Ar | Ar | 39,948 | ✅ | ✅ | 3,542 | 93,30 | Poling/PCO |
| C2H2 | C2H2 | 26,037 | ✅ | ✅ | 4,033 | 231,80 | Poling/PCO |
| C2H4 | C2H4 | 28,053 | ✅ | ✅ | 4,163 | 224,70 | Poling/PCO |
| C2H6 | C2H6 | 30,069 | ✅ | ✅ | 4,443 | 215,70 | Poling/PCO |
| C3H8 | C3H8 | 44,096 | ✅ | ✅ | 5,118 | 237,10 | Poling/PCO |
| C6H6 | C6H6 | 78,114 | ✅ | ✅ | 5,349 | 412,30 | Poling/PCO |
| CCl4 | CCl4 | 153,823 | ✅ | ✅ | 5,947 | 322,70 | Poling/PCO |
| CFCl3 | CFCl3 | 137,368 | ✅ | ⚠️ estimado | 5,290 | 341,40 | Poling σ=0,841Vc^⅓ |
| CH3CCl3 | CH3CCl3 | 133,403 | ✅ | ⚠️ estimado | 5,250 | 395,30 | Poling σ=0,841Vc^⅓ |
| CH4 | CH4 | 16,043 | ✅ | ✅ | 3,880 | 148,60 | Poling/PCO |
| Cl2 | Cl2 | 70,906 | ✅ | ✅ | 4,217 | 316,00 | Svehla |
| CO | CO | 28,010 | ✅ | ✅ | 3,690 | 91,70 | Poling/PCO |
| CO2 | CO2 | 44,010 | ✅ | ✅ | 3,941 | 195,20 | Poling/PCO |
| H2 | H2 | 2,016 | ✅ | ⚠️ quântico | 2,827 | 59,70 | Poling/PCO |
| H2O | H2O | 18,015 | ✅ | ⚠️ polar | 2,641 | 809,10 | Poling/PCO |
| H2S | H2S | 34,081 | ✅ | ⚠️ polar | 3,623 | 301,10 | Poling/PCO |
| HBr | HBr | 80,912 | ✅ | ✅ | 3,353 | 345,00 | Svehla |
| HCl | HCl | 36,461 | ✅ | ✅ | 3,339 | 344,70 | Svehla |
| He | He | 4,003 | ✅ | ⚠️ quântico | 2,551 | 10,22 | Poling/PCO |
| HI | HI | 127,912 | ✅ | ✅ | 4,211 | 288,70 | Svehla |
| I2 | I2 | 253,809 | ✅ | ✅ | 5,160 | 472,00 | Svehla |
| Kr | Kr | 83,798 | ✅ | ✅ | 3,655 | 178,90 | Poling/PCO |
| N2 | N2 | 28,013 | ✅ | ✅ | 3,798 | 71,40 | Poling/PCO |
| N2O | N2O | 44,013 | ✅ | ✅ | 3,828 | 232,40 | Poling/PCO |
| Ne | Ne | 20,180 | ✅ | ⚠️ quântico | 2,820 | 32,80 | Poling/PCO |
| NH3 | NH3 | 17,031 | ✅ | ⚠️ polar | 2,900 | 558,30 | Poling/PCO |
| **NO** | **NO** | 30,006 | ✅ | ⚠️ associante | 3,492 | 116,70 | Poling/PCO |
| O2 | O2 | 31,999 | ✅ | ✅ | 3,467 | 106,70 | Poling/PCO |
| SO2 | SO2 | 64,064 | ✅ | ⚠️ polar | 4,112 | 335,40 | Poling/PCO |
| Xe | Xe | 131,293 | ✅ | ✅ | 3,875 | 231,00 | Poling/PCO |

Os ⚠️ são **ressalvas de acurácia anotadas no próprio YAML** (LJ é aproximação para polares,
quânticos e associantes), não ausência de dados.

Um registro estendido opcional (`air_transport.yaml`) cobre 5 espécies (Ar, CO2, H2O, N2, O2)
com Tc/Pc/Vc/Zc/ω e coeficientes de referência **não consumidos pelo cálculo** — gancho declarado
para extensão a gás denso.

---

## 3. N2 ASSUMPTIONS

### 3.1 Busca textual

```
grep -rn -i "n2|nitrogen" src/statthermopy/transport/ --include=*.py
→ 0 ocorrências
```

Em todo o `src/`, as únicas menções a N2 são:

| Arquivo | Linha | Contexto | Classificação |
|---|---|---|---|
| `fluids.py` | 45 | `"N2": 0.78084` na composição do ar seco | **1 PHYSICALLY CORRECT** — N₂ é 78 % do ar |
| `core/molecule.py` | 287, 289 | exemplo em docstring | não é código |
| `validation/base.py` | 23 | exemplo em docstring | não é código |

**Nenhuma ocorrência de `property(species) = property(N2)` ou equivalente indireto.**

### 3.2 Teste de identidade numérica (§19 do pedido)

Comparação de cada uma das 29 demais espécies contra N₂ a 300 K, 101325 Pa:

| Propriedade | Espécies numericamente idênticas ao N₂ |
|---|---|
| μ | **0** de 29 |
| k | **0** de 29 |
| Pr | **0** de 29 |
| Sc | **0** de 29 |
| D_self | **0** de 29 |
| ρ | **0** de 29 |

Dispersão observada — incompatível com qualquer reaproveitamento:

| Propriedade | mínimo | máximo | razão |
|---|---|---|---|
| μ | 7,58×10⁻⁶ (C6H6) | 3,12×10⁻⁵ (Ne) | 4,1× |
| k | 2,54×10⁻³ (I2) | 1,74×10⁻¹ (H2) | **68,6×** |
| D_self | 1,93×10⁻⁶ (I2) | 1,68×10⁻⁴ (He) | **87,1×** |

**Coincidências aparentes que NÃO são fallback:**

* Pr = 0,66667 exato para Ar, He, Kr, Ne, Xe — é 4γ/(9γ−5) com γ = 5/3, o valor monoatômico
  exato de Eucken. Física correta.
* Cp ≈ 29,11 J/mol·K para N2, CO, NO, HCl, HBr, HI, H2 a 300 K — todas diatômicas com vibração
  congelada, logo 7R/2. Física correta.

### 3.3 Veredito da §3

**Não existe dependência de N₂ no módulo de transporte, nem direta nem indireta.** A hipótese
que motivou esta auditoria não se confirma.

---

## 4. PER-SPECIES TRANSPORT SUPPORT

Diagnóstico executado para as 30 espécies a T = 300 K, P = 101325 Pa (extrato; a tabela completa
está no script da §18 reproduzível):

| Gas | μ [Pa·s] | k [W/m/K] | Cp [J/mol/K] | Cv | ρ [kg/m³] | α [m²/s] | Pr | Fonte |
|---|---|---|---|---|---|---|---|---|
| He | 2,0233e-05 | 1,5761e-01 | 20,786 | 12,472 | 0,1626 | 1,867e-04 | 0,6667 | CE+LJ próprio |
| H2 | 8,8873e-06 | 1,7411e-01 | 29,101 | 20,786 | 0,0819 | 1,473e-04 | 0,7368 | CE+LJ próprio |
| N2 | 1,7698e-05 | 2,4960e-02 | 29,114 | 20,799 | 1,1380 | 2,111e-05 | 0,7369 | CE+LJ próprio |
| O2 | 2,0569e-05 | 2,5544e-02 | 29,345 | 21,031 | 1,2999 | 2,143e-05 | 0,7385 | CE+LJ próprio |
| CO2 | 1,5186e-05 | 1,6399e-02 | 37,131 | 28,817 | 1,7878 | 1,087e-05 | 0,7813 | CE+LJ próprio |
| H2O | 1,0757e-05 | 2,6203e-02 | 33,489 | 25,174 | 0,7318 | 1,926e-05 | 0,7632 | CE+LJ próprio |
| I2 | 1,3679e-05 | 2,5391e-03 | 36,719 | 28,405 | 10,3102 | 1,702e-06 | 0,7794 | CE+LJ próprio |
| CCl4 | 9,8032e-06 | 5,9572e-03 | 83,083 | 74,768 | 6,2486 | 1,765e-06 | 0,8888 | CE+LJ próprio |

**Nenhuma célula MISSING. Nenhuma célula N2 FALLBACK.**

### Dependência de temperatura (§20)

Nas sete temperaturas 200/250/300/400/500/750/1000 K:

* **μ**: varia entre 2,84× (He) e 4,85× (NH3). **0 correlações constantes.**
* **k**: varia entre 2,84× (He) e 16,01× (C6H6). **0 correlações constantes.**
* **Cp**: constante para Ar, He, Kr, Ne, Xe (5 de 30) — 5R/2 exato dos monoatômicos, **física
  correta**, não defeito.

---

## 5. EXISTING MIXTURE SUPPORT

| Nível | Suportado | Evidência |
|---|---|---|
| Gás puro | ✅ | `TransportCalculator`, 30 espécies |
| Binária | ✅ | testado N2/O2 e He/Xe |
| Ternária | ✅ | testado N2/O2/Ar |
| Quaternária | ✅ | ar seco |
| Multicomponente arbitrário | ✅ | testado com 5, 8 e **15** componentes |

**Busca por limites fixos** (`len(...) == 2`, `n_components`, `binary only`, …): **nenhum
encontrado.** Uma única formulação trata qualquer N.

Resultados medidos a 300 K, 1 atm:

| Caso | n | μ [Pa·s] | k [W/m/K] | M [g/mol] | Pr |
|---|---:|---|---|---:|---|
| N2 puro | 1 | 1,76985e-05 | 2,49598e-02 | 28,013 | 0,7369 |
| N2/O2 | 2 | 1,91341e-05 | 2,52725e-02 | 30,006 | 0,7375 |
| **He/Xe** | 2 | **2,68317e-05** | 8,10887e-03 | 67,648 | 1,0167 |
| N2/O2/Ar | 3 | 1,96580e-05 | 2,32187e-02 | 31,596 | 0,7374 |
| ar seco | 4 | 1,83510e-05 | 2,49924e-02 | 28,966 | 0,7374 |
| ar úmido | 5 | 1,81148e-05 | 2,50289e-02 | 28,625 | 0,7388 |
| leve (H2/He/CH4/NH3) | 4 | 1,24167e-05 | 5,99686e-02 | 6,919 | 0,8550 |
| pesada (CCl4/I2/C6H6/SO2) | 4 | 1,11886e-05 | 4,49891e-03 | 159,701 | 1,0154 |
| 8 componentes | 8 | 1,89245e-05 | 1,63238e-02 | 34,301 | 1,0034 |
| **15 componentes** | 15 | 1,78066e-05 | 1,61116e-02 | 37,440 | 0,8786 |

> **Prova de que Wilke está de fato ativo:** He/Xe dá μ_mix = 2,683×10⁻⁵ — **maior que a de
> qualquer um dos puros** (He 2,02×10⁻⁵, Xe 2,52×10⁻⁵). Esse máximo não monotônico para massas
> muito díspares é a assinatura clássica da regra de Wilke e é impossível de obter por média
> ponderada ou por qualquer atalho baseado em N₂.

A composição realmente governa o resultado (ar úmido, 300 K, 1 atm):

| x(H₂O) | μ [Pa·s] | k [W/m/K] | M [g/mol] | Pr |
|---|---|---|---|---|
| 0,001 | 1,83434e-05 | 2,49937e-02 | 28,955 | 0,73741 |
| 0,010 | 1,82753e-05 | 2,50052e-02 | 28,856 | 0,73784 |
| 0,030 | 1,81239e-05 | 2,50309e-02 | 28,637 | 0,73879 |
| 0,100 | 1,75939e-05 | 2,51208e-02 | 27,871 | 0,74202 |

μ cai, k sobe, M cai, Pr sobe — todas as tendências corretas.

---

## 6. PHYSICAL VALIDITY

| Aproximação | Classificação | Observação |
|---|---|---|
| Chapman–Enskog 1ª ordem | **1 PHYSICALLY CORRECT** | limite de gás diluído, exato nesse regime |
| Neufeld Ω^(l,s)* | **1** | ajuste a 0,1 % das integrais exatas de LJ |
| Eucken para k | **2 ACCEPTABLE APPROXIMATION** | erro de 4–12 % em poliatômicas; Eucken modificado seria melhor |
| Wilke para μ_mix | **1** | padrão de referência, erro típico < 2 % |
| Mason–Saxena com φ de Wilke | **2** | usa razão μ em vez de razão k; documentado no código |
| Blanc para D_i,mix | **2** | rigoroso só para traço diluído |
| LJ para polares (H₂O, NH₃, SO₂, H₂S) | **2** | anotado por espécie; erro de k até 34 % (H₂O) |
| LJ para quânticos (H₂, He, Ne) | **2** | anotado por espécie |
| σ estimado (CFCl3, CH3CCl3) | **2** | correlação de Poling σ = 0,841 Vc^⅓, anotado |
| Gás ideal (Z=1, μ_JT=0) | **1** | exato para o modelo adotado |
| `trace="H2O"` como padrão de D_eff | **3 TEMPORARY SIMPLIFICATION** | ver §7 |

### Acurácia medida contra literatura (300 K, 1 atm)

| Gas | μ motor | μ exp | dif | k motor | k exp | dif |
|---|---|---|---|---|---|---|
| N2 | 1,7698e-05 | 1,78e-05 | −0,6 % | 2,4960e-02 | 0,0260 | −4,0 % |
| O2 | 2,0569e-05 | 2,07e-05 | −0,6 % | 2,5544e-02 | 0,0266 | −4,0 % |
| Ar | 2,2820e-05 | 2,27e-05 | +0,5 % | 1,7811e-02 | 0,0177 | +0,6 % |
| CO2 | 1,5186e-05 | 1,50e-05 | +1,2 % | 1,6399e-02 | 0,0166 | −1,2 % |
| H2 | 8,8873e-06 | 8,96e-06 | −0,8 % | 1,7411e-01 | 0,187 | −6,9 % |
| He | 2,0233e-05 | 1,99e-05 | +1,7 % | 1,5761e-01 | 0,155 | +1,7 % |
| CH4 | 1,0495e-05 | 1,12e-05 | −6,3 % | 3,0160e-02 | 0,0343 | −12,1 % |
| **H2O** | 1,0757e-05 | 9,80e-06 | **+9,8 %** | 2,6203e-02 | 0,0196 | **+33,7 %** |
| NH3 | 1,0296e-05 | 1,01e-05 | +1,9 % | 2,7761e-02 | 0,0247 | +12,4 % |
| CO | 1,7684e-05 | 1,78e-05 | −0,7 % | 2,4952e-02 | 0,0250 | −0,2 % |
| | | **médio 2,4 %** | | | **médio 7,7 %** | |

Ar seco como mistura: μ −0,8 %, k −5,0 %, Pr +4,3 % contra os valores tabelados.

**O erro é dominado pela correlação de Eucken e pela aproximação LJ para polares — não por
qualquer reaproveitamento de espécie.**

---

## 7. MISSING PHYSICS / DEFEITOS ENCONTRADOS

### D-1 · `NO.yaml` é lido como booleano — **defeito real**

`name: NO` e `formula: NO` sem aspas. YAML 1.1 (PyYAML) interpreta `NO` como **booleano falso**:

```python
yaml.safe_load(...)["name"]   # → False (bool)
get("NO").name                 # → 'False'
get("NO").formula              # → 'False'
```

A chave do registro vem do nome do arquivo, então o *lookup* funciona e os **números estão
corretos**. Mas o rótulo da espécie é a string `"False"` em toda saída — e
`MixtureTransportProperties.components` é indexado por `mol.name`, de modo que uma mistura
contendo NO reporta a componente sob a chave `"False"`.

**Severidade: ALTA** (corrupção silenciosa de rótulo). **Correção: uma linha** — aspas no YAML.

### D-2 · Fração molar zero quebra a mistura — **defeito real**

```python
IdealGasMixture.from_names({"N2": 1.0, "H2O": 0.0}).compute(State(T=300, P=101325))
→ ValueError: Pressure P must be > 0 Pa
```

`mixture.py:169` monta `State(T=T, P=Pi)` com pressão parcial `Pi = x_i·P`; com `x_i = 0` isso é
zero e `State` rejeita. `x = 1e-12` funciona.

O caminho oficial `AirTransport.humid(RH=0)` **não** é afetado (remove a componente). Atinge quem
constrói a mistura diretamente — o caso natural de varrer composição de 0 a 1.

**Severidade: MÉDIA.** **Correção: pular componentes com x_i = 0** no laço de `compute`.

### D-3 · `trace="H2O"` é padrão air-centric — **simplificação**

`D_eff`, e portanto `Sc` e `Le`, são definidos como difusão do traçador na mistura, com padrão
H₂O **mesmo em misturas sem água**:

| Mistura pesada (CCl4/I2/C6H6/SO2) | D_eff | Sc | Le |
|---|---|---|---|
| `trace="H2O"` (padrão) | 8,4308e-06 | 0,2046 | 0,2015 |
| `trace="CCl4"` | 2,4598e-06 | 0,7011 | 0,6905 |

O parâmetro **é configurável**, então não é um hardcode — mas o padrão faz Sc/Le de uma mistura
arbitrária se referirem a vapor d'água difundindo nela, o que raramente é o pretendido.

**Severidade: MÉDIA** (semântica silenciosa). **Correção: exigir `trace` explícito quando a
espécie não estiver na mistura, ou reportar `D_im` por componente como grandeza principal.**

### D-4 · `MixtureTransportCalculator` só é alcançável sob `.air`

`from statthermopy.transport.air import MixtureTransportCalculator` — o motor é totalmente
genérico, mas o namespace sugere o contrário. Quem quiser uma mistura CH₄/H₂ não procuraria ali.
**Severidade: BAIXA** (descoberta, não física).

### D-5 · Física ausente (não são defeitos, são limites declarados)

* **Eucken modificado** (Mason–Monchick) para poliatômicas — reduziria o erro de k de 12 % para
  ~3 % em CH₄/NH₃.
* **Maxwell–Stefan** — hoje há apenas Blanc (mixture-averaged).
* **Correção de gás denso** (Enskog / estados correspondentes) — Tc/Pc/ω já estão armazenados
  como gancho, mas nenhum caminho os consome.
* **Difusão térmica** (efeito Soret) — ausente; relevante só para misturas H₂/pesado.
* **Potencial de Stockmayer** para polares — substituiria LJ em H₂O/NH₃/SO₂/H₂S.

---

## 8. RECOMMENDED MIXTURE MODEL

O que já existe **é** o modelo recomendado para o escopo de gás diluído:

| Grandeza | Formulação atual | Recomendação |
|---|---|---|
| μ_mix | Wilke | **manter** |
| k_mix | Mason–Saxena (φ de Wilke) | manter; opcionalmente φ com razão k |
| Cp_mix, Cv_mix | Σ x_i Cp_i da função de partição | **manter** |
| ρ_mix | P·M_mix/(RT) | **manter** |
| α_mix | k/(ρc_p) | **manter** |
| D_ij | Chapman–Enskog + Lorentz–Berthelot | **manter** |
| D_i,mix | Blanc | manter; expor D_im por componente |
| Pr, Sc, Le | da mistura | manter; corrigir semântica do traçador (D-3) |

**Maxwell–Stefan não é recomendado agora.** Blanc coincide com M-S no limite de traço diluído e
o erro cresce só quando duas espécies não traço têm difusividades muito diferentes *e* gradientes
opostos. O custo é resolver um sistema n×n por ponto. Só se justifica se o STATTHERMOPY passar a
alimentar CFD reativo multicomponente com espécies em fração comparável.

---

## 9. N2 FALLBACK MATRIX

| Gas | μ | k | Cp | Cv | D | Dependência de N₂ |
|---|---|---|---|---|---|---|
| Ar | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| C2H2 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| C2H4 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| C2H6 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| C3H8 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| C6H6 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| CCl4 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| CFCl3 | próprio (σ est.) | próprio | próprio | próprio | próprio | **nenhuma** |
| CH3CCl3 | próprio (σ est.) | próprio | próprio | próprio | próprio | **nenhuma** |
| CH4 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| Cl2 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| CO | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| CO2 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| H2 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| H2O | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| H2S | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| HBr | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| HCl | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| He | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| HI | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| I2 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| Kr | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| N2 | próprio | próprio | próprio | próprio | próprio | — (é o N₂) |
| N2O | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| Ne | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| NH3 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| NO | próprio | próprio | próprio | próprio | próprio | **nenhuma** (rótulo quebrado, D-1) |
| O2 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| SO2 | próprio | próprio | próprio | próprio | próprio | **nenhuma** |
| Xe | próprio | próprio | próprio | próprio | próprio | **nenhuma** |

**30 de 30 sem qualquer dependência de N₂.**

---

## 10. BINARY / TERNARY / MULTICOMPONENT ASSESSMENT

O código é **verdadeiramente genérico até N arbitrário**. Não há caminho separado para binário,
ternário ou quaternário: `wilke_viscosity`, `mason_saxena_conductivity` e `blanc_diffusion`
recebem listas de comprimento n e iteram. A matriz D_ij é montada n×n sem caso especial.

Único limite prático é o custo O(n²) da matriz de difusão (§13), não a formulação.

---

## 11. THERMODYNAMIC CONSISTENCY

A cadeia `species → composition → thermodynamics → transport` usa **uma única composição**:
`MixtureTransportCalculator.compute` lê `mixture.x` uma vez e a mesma lista alimenta
`mix.compute()` (termodinâmica) e as regras de mistura (transporte).

Identidades verificadas em três misturas muito diferentes (ar seco, He/Xe, pesada):

| Identidade | Resíduo máximo |
|---|---|
| Cp_mix − Cv_mix − R | 1,07×10⁻¹⁴ |
| R_mix − R_u/M_mix | 0 |
| ρ_mix − P·M_mix/(RT) | 0 |
| Σ x_i − 1 | 1,11×10⁻¹⁶ |
| a − √(γ R_mix T) | 0 |
| M_mix − Σ x_i M_i | 0 |
| Pr − μc_p/k | 0 |
| Le − α/D_eff | 5,6×10⁻¹⁷ |

Conversão base mássica → molar verificada: He/Xe 50/50 em massa dá x = 0,9704/0,0296 em mol
(correto, He é 21× mais leve) e M_mix = 7,768 g/mol contra 67,648 na base molar.

**Nenhuma inconsistência encontrada.**

---

## 12. CFD IMPACT

Como a auditoria **não encontrou dependência de N₂**, não há correção de transporte a propagar
para momentum, energia ou espécies. O que existe é um limite de acurácia herdado do modelo:

| Equação | Coeficiente | Erro atual | Impacto |
|---|---|---|---|
| Momentum ∇·[μ(∇u+∇uᵀ)] | μ_mix | ~1–2 % (não polar) | desprezível; Re dentro de 2 % |
| Energia ∇·(k∇T) | k_mix | 4–12 % (Eucken) | **dominante**; afeta Nu e camada térmica |
| Espécies ∇·(ρD∇Y) | D_i,mix | ~5 % (Blanc) | moderado |
| — | Pr | +4,3 % no ar | propaga para correlações de Nu |

**A prioridade para CFD é o Eucken modificado, não a independência de espécies (que já existe).**
Para escoamento com vapor d'água, o erro de 34 % em k(H₂O) é o item mais sério.

---

## 13. CPU/GPU PERFORMANCE IMPACT

Custo medido por ponto de avaliação:

| n componentes | tempo | trabalho |
|---:|---:|---|
| 2 | 0,9 ms | 4 D_ij + 2 CE |
| 4 | 16,6 ms | 16 D_ij + 4 CE |
| 8 | 23,8 ms | 64 D_ij + 8 CE |
| 15 | 76,7 ms | 225 D_ij + 15 CE |
| 30 | 80,7 ms | 900 D_ij + 30 CE |

**Adequado para uma calculadora de propriedades; inadequado para CFD célula a célula.** A 81 ms
por ponto, um milhão de células levaria ~22 h por avaliação.

A implementação é escalar e em Python puro: laços sobre espécies **e** uma chamada
`TransportCalculator` (que reavalia a função de partição) por espécie por ponto.

Estratégia recomendada, se o alvo for CFD:

1. **Pré-computar** μ_i(T) e k_i(T) em tabela por espécie e interpolar — remove a função de
   partição do laço interno.
2. **Vetorizar** φ_ij como operação de broadcasting `(..., n, n)` sobre o array de células.
3. **Cachear** a matriz D_ij·P, que depende só de T (D ∝ 1/P é fator escalar).
4. Manter laços sobre espécies (n ≤ 30, barato); eliminar laços sobre células.

Nada disso exige mudar a física — só a organização dos dados.

---

## 14. IMPLEMENTATION PLAN

### CRITICAL

| # | Ação | Esforço |
|---|---|---|
| C-1 | Aspas em `NO.yaml` (`name: "NO"`, `formula: "NO"`) + teste que varra os 30 YAMLs verificando `mol.name == chave` | 5 min |

### HIGH

| # | Ação | Esforço |
|---|---|---|
| H-1 | Pular componentes com `x_i == 0` em `IdealGasMixture.compute` | ~10 linhas |
| H-2 | Semântica do traçador (D-3): expor `D_im` por componente e exigir `trace` explícito quando ausente da mistura | ~20 linhas |

### MEDIUM

| # | Ação | Esforço |
|---|---|---|
| M-1 | Reexportar `MixtureTransportCalculator` em `statthermopy.transport` | 2 linhas |
| M-2 | Eucken modificado (Mason–Monchick) para poliatômicas | ~30 linhas |
| M-3 | φ_ij de Mason–Saxena com razão k em vez de razão μ | ~15 linhas |

### OPTIONAL

| # | Ação |
|---|---|
| O-1 | Potencial de Stockmayer para polares (exige momento dipolar por espécie) |
| O-2 | Caminho vetorizado para CFD (§13) |
| O-3 | Correção de gás denso consumindo Tc/Pc/ω já armazenados |
| O-4 | Maxwell–Stefan — **só se** surgir demanda de CFD reativo multicomponente |

---

## 15. FILES / FUNCTIONS TO MODIFY

| Item | Arquivo | Local |
|---|---|---|
| C-1 | `src/statthermopy/database/data/NO.yaml` | linhas 1–2 |
| C-1 (teste) | `tests/test_database.py` | novo teste |
| H-1 | `src/statthermopy/mixture.py` | `IdealGasMixture.compute`, linha 169 |
| H-2 | `src/statthermopy/transport/air/mixture_transport.py` | `MixtureTransportCalculator.__init__` e `_trace_diffusivity` |
| M-1 | `src/statthermopy/transport/__init__.py` | `__all__`, linha 46 |
| M-2 | `src/statthermopy/transport/transport.py` | `TransportCalculator.conductivity`, linha ~183 |
| M-3 | `src/statthermopy/transport/air/mixture_transport.py` | `mason_saxena_conductivity`, linha ~140 |

---

## CRITÉRIO FINAL

> **O STATTHERMOPY possui atualmente propriedades de transporte independentes e fisicamente
> consistentes para todos os gases?**
>
> ## **SIM**
>
> As 30 espécies (não 23) têm σ e ε/k próprios e distintos, e μ, k, D, Pr, Sc, Le são calculados
> por Chapman–Enskog a partir desses parâmetros mais C_v/γ da função de partição. Zero espécies
> compartilham qualquer valor com o N₂.

> **Suporta misturas binárias?** ## **SIM**
> **Ternárias?** ## **SIM**
> **Multicomponentes arbitrárias?** ## **SIM** — testado com 15 componentes; nenhum limite fixo
> no código.

### Quantificação

| Métrica | Valor |
|---|---|
| Gases com propriedades próprias | **30 de 30 (100 %)** |
| Gases usando alguma propriedade de N₂ | **0** |
| Gases incompletos (sem LJ) | **0** |
| Gases com ressalva de acurácia anotada | 9 (polares, quânticos, σ estimado) |
| Propriedades que precisam de correção | **0 por dependência de espécie**; 2 defeitos de robustez (D-1, D-2) e 1 de semântica (D-3) |
| Pares binários calculáveis | **435 de 435**, nenhum armazenado |

### Correções indispensáveis antes de chamar o módulo de "verdadeiramente multicomponente"

**Nenhuma, quanto à física multicomponente** — ela já está correta e é genérica.

Indispensáveis quanto a **robustez e clareza**:

1. **C-1** — `NO.yaml` (rótulo corrompido em qualquer mistura contendo NO).
2. **H-1** — fração molar zero (impede varredura de composição de 0 a 1).
3. **H-2** — semântica do traçador de Sc/Le fora do contexto de ar.

### Conclusão

A hipótese que motivou esta auditoria — de que a infraestrutura termodinâmica multiespécie estaria
mascarando um modelo de transporte baseado em N₂ — **não se confirma**. O módulo de transporte é
genuinamente multiespécie e multicomponente. Os limites reais são de **acurácia de modelo**
(Eucken para poliatômicas, LJ para polares) e de **desempenho para CFD**, não de arquitetura.
