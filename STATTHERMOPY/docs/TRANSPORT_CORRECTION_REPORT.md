# Correções do módulo de transporte

**Data:** 2026-09-10 · **Base:** `docs/TRANSPORT_MIXTURE_AUDIT.md` · **Escopo:** correções, acurácia
e desempenho, preservando a arquitetura multicomponente.

---

## EXECUTIVE SUMMARY

A auditoria concluiu que a arquitetura estava correta e que **não havia dependência de N₂**. Nada
disso foi tocado: Wilke, Mason–Saxena, Blanc, D_ij dinâmico e a generalidade em n componentes
seguem exatamente como estavam, agora protegidos por testes de regressão que falham se forem
substituídos.

| | Antes | Depois |
|---|---|---|
| `get("NO").name` | `"False"` (booleano YAML) | `"NO"` |
| Fração molar zero | `ValueError` | equivalente exato a omitir a espécie |
| `Sc`/`Le`/`D_eff` genéricos | H₂O implícito, mesmo ausente | por espécie; traçador explícito |
| Erro médio de μ (20 espécies) | 2,4 % | **1,7 %** |
| Erro de μ(H₂O) a 300 K | +9,8 % | **+2,3 %** |
| Erro de k(H₂O) a 300 K | +33,7 % | **+24,5 %** |
| k(H₂O) vs k(ar) a 300 K | invertido (H₂O acima) | ordenamento correto |
| Mistura de 30 componentes | 81 ms/ponto | **13,2 ms/ponto** |
| Caminho vetorizado | inexistente | **310 000 pontos/s** (5 espécies) |
| Testes | 545 | **646** |

Um resultado **negativo** também foi estabelecido por medição: o Eucken modificado, hipótese
natural para melhorar k, **piora** o conjunto (6,6 % → 10,3 %). Não foi adotado. Ver §THERMAL
CONDUCTIVITY ACCURACY.

---

## FIX D-1 — NO YAML

**Causa.** `NO.yaml` trazia `name: NO` e `formula: NO` sem aspas. YAML 1.1, que o PyYAML
implementa, resolve `NO` como o booleano falso:

```python
yaml.safe_load(...)["name"]   # -> False (bool)
get("NO").name                 # -> 'False'
```

A chave do registro vem do nome do arquivo, então o *lookup* funcionava e os números estavam
corretos — mas o rótulo da espécie era a string `"False"` em toda saída, incluindo as chaves
por componente de um relatório de transporte de mistura.

**Varredura.** Todos os YAMLs das 30 espécies foram verificados contra a lista completa de
escalares que o YAML 1.1 converte (`y`, `n`, `yes`, `no`, `on`, `off`, `true`, `false`, `null` e
variantes de capitalização). Foram encontradas **três** ocorrências, não uma:

| Arquivo | Campo | Antes | Depois |
|---|---|---|---|
| `database/data/NO.yaml` | `name` | `False` | `"NO"` |
| `database/data/NO.yaml` | `formula` | `False` | `"NO"` |
| `validation/data/NO.yaml` | `species` | `False` | `"NO"` |

A terceira não tinha sido vista na auditoria. `He.yaml` traz `rotational_temperatures: None`, que
é intencional (monoatômico) e ficou como está.

**Testes.** Quatro, em `tests/test_database.py`:

* `test_no_is_nitric_oxide_not_a_boolean` — o caso específico;
* `test_species_names_are_strings_not_yaml_booleans` — as 30 espécies, com `mol.name == chave`;
* `test_every_database_yaml_has_string_name_and_formula` — varre os arquivos crus;
* `test_validation_reference_species_names_are_strings` — idem para a validação.

Uma espécie futura chamada `ON`, `OFF` ou `Y` não passa despercebida.

---

## FIX D-2 — ZERO MOLE FRACTIONS

**Causa.** `IdealGasMixture.compute` avalia cada componente na sua pressão parcial
`P_i = x_i·P`; com `x_i = 0` isso é zero e `State` rejeita pressão não positiva.

**Correção.** Em `IdealGasMixture.__init__`, uma espécie com fração zero é tratada como
**ausente**, não como infinitamente diluída. A justificativa é física: sua pressão parcial é zero e
sua entropia molar diverge, mas toda contribuição extensiva `x_i·(...)` tem limite zero —
inclusive `−R x ln x`. Nada é substituído por epsilon; qualquer fração estritamente positiva é
mantida, por menor que seja.

```python
self.inactive = tuple(mol.name for mol, v in components.items() if v <= 0.0)
active = {mol: v for mol, v in components.items() if v > 0.0}
```

Simultaneamente (§3 do pedido): frações negativas são rejeitadas com mensagem nomeando a espécie;
exige-se soma positiva; não se exige `x_i > 0` para todas.

**Resultado.** `{"N2": 1.0, "H2O": 0.0}` é **numericamente idêntico** a `{"N2": 1.0}` — diferença
exatamente `0.0`, não dentro de tolerância:

| Grandeza | com H₂O a zero | sem H₂O | diferença |
|---|---|---|---|
| μ | 1,769845471213e-05 | 1,769845471213e-05 | 0,0 |
| k | 2,495975904901e-02 | 2,495975904901e-02 | 0,0 |
| ρ | 1,137959996277e+00 | 1,137959996277e+00 | 0,0 |
| M | 2,801340000000e-02 | 2,801340000000e-02 | 0,0 |

**Continuidade da varredura** de x(H₂O) = 0 a 1:

| x(H₂O) | μ [Pa·s] | salto vs anterior |
|---|---|---|
| 0 | 1,835097853e-05 | — |
| 1e-15 | 1,835097853e-05 | 6,8e-21 |
| 1e-12 | 1,835097853e-05 | 7,6e-18 |
| 1e-9 | 1,835097852e-05 | 7,6e-15 |
| 1e-6 | 1,835097096e-05 | 7,6e-12 |
| 0,01 | 1,827530057e-05 | 7,6e-08 |
| 1,0 | 1,075733204e-05 | — |

Sem descontinuidade em zero. Composição esparsa (`{"N2":0.78,"O2":0.21,"Ar":0.01,"H2O":0.0,
"CO2":0.0}`) funciona sem poda manual. Testado também em base mássica.

---

## FIX D-3 — TRACE SPECIES

**Mudança conceitual.** Difusão não é um escalar da mistura. `D_i,m`, e portanto `Sc_i` e `Le_i`,
são definidos **por espécie**. `α` e `Pr` continuam sendo escalares da mistura, e a API agora
separa os dois tipos.

| Antes | Depois |
|---|---|
| `trace="H2O"` implícito | `trace=None` genérico |
| `Sc`, `Le`, `D_eff` escalares sempre | `Sc_i`, `Le_i`, `D_im` por espécie, sempre |
| H₂O difundindo em mistura sem água | `Sc`/`Le`/`D_eff` são `None` sem traçador |
| — | `trace_species` registra a escolha |

**API nova**, preservando a existente:

```python
r = MixtureTransportCalculator(mix).compute(state)
r.effective_diffusivities()   # {"CCl4": 2.46e-06, "I2": 2.40e-06, ...}
r.schmidt_numbers()           # {"CCl4": 0.7011, ...}
r.lewis_numbers()
r.schmidt_number("CCl4")      # 0.7011
r.schmidt_number("H2O")       # KeyError, com a lista de componentes na mensagem
```

Pedir uma espécie ausente falha com mensagem explícita, em vez de devolver silenciosamente um
número sobre outra substância.

**Compatibilidade.** `AirTransport.dry()` e `.humid()` passam `trace="H2O"` explicitamente, onde é
a escolha fisicamente natural. CLI, GUI, gráficos e exportadores não mudaram de comportamento.

**O caso que motivou a correção**, mistura CCl₄/I₂/C₆H₆/SO₂:

| | D_eff | Sc | Le |
|---|---|---|---|
| antes (H₂O implícito) | 8,4308e-06 | 0,2046 | 0,2015 |
| agora, genérico | `None` | `None` | `None` |
| agora, `Sc_i` real | — | CCl₄ 0,7011 · I₂ 0,7191 · C₆H₆ 0,5896 · SO₂ 0,4210 | — |

---

## THERMAL CONDUCTIVITY ACCURACY

### Separação obrigatória: erro da propriedade pura vs erro da regra de mistura

Feita antes de qualquer alteração, como exige o §7 do pedido. A regra de mistura de ar seco erra
−5,0 % em k enquanto os componentes puros erram −4,0 % (N₂, O₂) e −1,2 % (CO₂). **O erro é
predominantemente da propriedade pura**, não de Mason–Saxena. Mason–Saxena foi preservado.

### Resultado negativo: o Eucken modificado piora

Hipótese testada: substituir o Eucken simples por
`k = (μ/M)[(5/2)C_v,trans + (ρD/μ)C_v,int]`, com `ρD/μ = (6/5)Ω^(2,2)*/Ω^(1,1)*`.

| Espécie | Eucken | Modificado |
|---|---:|---:|
| H₂ | −6,9 % | **−0,7 %** |
| CH₄ | −12,1 % | **−3,2 %** |
| N₂ | −4,0 % | +2,4 % |
| CO₂ | −1,2 % | +9,6 % |
| NH₃ | +12,4 % | **+30,1 %** |
| H₂S | +18,7 % | **+31,6 %** |
| H₂O | +33,7 % | **+56,1 %** |
| **médio (20 espécies)** | **6,6 %** | **10,3 %** |

Ajuda hidrocarbonetos e H₂, arruína os polares, e piora o conjunto em 56 %. **Não adotado.** A
medição precedeu a decisão, conforme o §7 exige.

### O que foi adotado: potencial de Stockmayer para espécies polares

Diagnóstico de k(H₂O), decompondo a contribuição de cada fator:

| Fonte do erro | Contribuição |
|---|---|
| μ (parâmetros LJ esféricos numa molécula com 1,85 D) | ~10 pontos percentuais |
| Eucken (modos internos de molécula polar) | ~24 pontos percentuais |

Confirmado usando μ **experimental** dentro do Eucken: o erro de k caía de 33,7 % para ~22 %, ou
seja, dois terços não vinham de μ.

Para a parcela de μ, a correção correta e de primeiros princípios é o **potencial de Stockmayer**
— LJ 12-6 mais um dipolo pontual — com a correção de Brokaw (1969) às integrais de colisão:

> Ω*(T*, δ) = Ω*_LJ(T*) + c·δ²/T* , com δ = μ_D²/(2εσ³), c = 0,2 (Ω^(2,2)*) e 0,19 (Ω^(1,1)*)

O momento de dipolo é uma **constante molecular**, não dado de propriedade: o núcleo permanece de
primeiros princípios.

**Implementação por metadados, não por `if species ==`** (§11 do pedido): um bloco opcional
`stockmayer` no YAML da espécie, lido para a dataclass `Stockmayer`. O motor consulta
`molecule.stockmayer`; quando ausente usa LJ com δ = 0, que é **exatamente** a expressão anterior.
Verificado: as 29 espécies não polares são idênticas bit a bit ao caminho LJ.

```yaml
stockmayer:
  sigma_angstrom: 2.52
  epsilon_over_k: 775.0
  dipole_debye: 1.85
  reference: "Brokaw (1969); Poling, Prausnitz & O'Connell Table 9-1."
```

**Só H₂O recebeu o bloco**, porque só nele a evidência sustenta a mudança:

| Espécie | δ | μ: erro LJ | μ: erro Stockmayer | veredito |
|---|---:|---:|---:|---|
| H₂O | 1,00 | +9,8 % | **+2,3 %** | adotado |
| NH₃ | 0,70 | +1,9 % | +2,1 % | não adotado — sem ganho |
| SO₂ | 0,42 | +1,3 % | +0,7 % | não adotado — ganho marginal |
| H₂S | 0,20 | +2,3 % | +2,4 % | não adotado — sem ganho |

Validado em faixa de temperatura, não em um ponto (§8): erro médio de μ(H₂O) em 373–1000 K cai de
4,9 % para 3,0 %.

### Correção qualitativa colateral

Com os parâmetros LJ o motor dava k(H₂O) = 0,0262 W/m·K a 300 K, **acima** do ar (0,0250) — e a
referência IAPWS de gás diluído dá 0,0186, **abaixo**. O ordenamento estava invertido. Com
Stockmayer, k(H₂O) = 0,0244 < k(ar), e o ordenamento passa a ser o correto. Consequência prática:
adicionar vapor d'água ao ar a 300 K agora **reduz** a condutividade da mistura, como deve.

### Erros finais, por espécie, ordenados pelo erro de k

| Species | Classe | Polaridade | erro μ | erro k | Limitação do modelo |
|---|---|---|---:|---:|---|
| H2O | poliatômico | polar | +2,3 % | +24,5 % | Eucken em molécula polar |
| H2S | poliatômico | polar | +2,3 % | +18,7 % | Eucken em molécula polar |
| NH3 | poliatômico | polar | +1,9 % | +12,4 % | Eucken em molécula polar |
| CH4 | poliatômico | não polar | −6,3 % | −12,1 % | Eucken (modos internos) |
| C2H6 | poliatômico | não polar | +0,6 % | −7,2 % | Eucken (modos internos) |
| H2 | diatômico | não polar | −0,8 % | −6,9 % | Eucken; fluido quântico |
| Cl2 | diatômico | não polar | −0,9 % | −6,8 % | Eucken (modos internos) |
| Xe | monoatômico | não polar | +8,6 % | +5,9 % | parâmetros LJ |
| C2H4 | poliatômico | não polar | −0,8 % | −5,5 % | Eucken (modos internos) |
| SO2 | poliatômico | polar | +1,3 % | +4,9 % | Eucken em molécula polar |
| N2 | diatômico | não polar | −0,6 % | −4,0 % | Eucken (modos internos) |
| O2 | diatômico | não polar | −0,6 % | −4,0 % | Eucken (modos internos) |
| N2O | poliatômico | fracamente polar | −0,6 % | −3,9 % | Eucken (modos internos) |
| Ne | monoatômico | não polar | −1,8 % | −1,7 % | nenhuma (Eucken exato p/ γ=5/3) |
| He | monoatômico | não polar | +1,7 % | +1,7 % | nenhuma |
| CO2 | poliatômico | não polar | +1,2 % | −1,2 % | Eucken (modos internos) |
| Ar | monoatômico | não polar | +0,5 % | +0,6 % | nenhuma |
| Kr | monoatômico | não polar | −0,6 % | −0,4 % | nenhuma |
| NO | diatômico | fracamente polar | −0,3 % | +0,4 % | Eucken (modos internos) |
| CO | diatômico | fracamente polar | −0,7 % | −0,2 % | Eucken (modos internos) |

**Médias:** μ = **1,7 %** (era 2,4 %) · k = **6,2 %** (era 6,6 %).
Por grupo: k dos 4 polares = 15,2 % · k das 16 demais = **3,9 %**.

Uma única formulação genérica **não** é adequada para todas: os monoatômicos são exatos, os não
polares ficam em ~4 %, e os polares concentram o erro residual.

---

## H2O TRANSPORT

Auditoria específica, como pede o §9, contra a IAPWS no limite de gás diluído:

| T [K] | μ motor | μ ref | erro | k motor | k ref (IAPWS) | erro |
|---|---|---|---:|---|---|---:|
| 300 | 1,0021e-05 | 9,80e-06 | +2,3 % | 0,024408 | 0,018563 | +31,5 % |
| 373,15 | 1,2580e-05 | 1,20e-05 | +4,8 % | 0,030951 | 0,024156 | +28,1 % |
| 400 | 1,3543e-05 | 1,32e-05 | +2,6 % | 0,033484 | 0,026431 | +26,7 % |
| 500 | 1,7223e-05 | 1,73e-05 | −0,4 % | 0,043510 | 0,035780 | +21,6 % |
| 600 | 2,1008e-05 | 2,14e-05 | −1,8 % | 0,054361 | 0,046276 | +17,5 % |
| 800 | 2,8669e-05 | 2,98e-05 | −3,8 % | 0,077994 | 0,069833 | +11,7 % |
| 1000 | 3,6177e-05 | 3,79e-05 | −4,5 % | 0,103469 | 0,095805 | +8,0 % |

**Origem quantificada do resíduo em k:** o erro **decai monotonicamente com a temperatura**, de
+31 % a 300 K para +8 % a 1000 K. Essa é a assinatura da relaxação rotacional: numa molécula
polar, a troca ressonante de energia rotacional é rápida a baixa temperatura e o Eucken — que
supõe energia interna transportada à taxa de difusão — deixa de valer. Não é μ (já corrigido), não
são C_p nem os graus internos (verificados na auditoria do H₂O), não é a referência (IAPWS de gás
diluído).

**O que resolveria:** Mason–Monchick com número de colisão rotacional Z_rot. Não implementado
porque Z_rot é ele próprio incerto e dependente de T, e ajustá-lo para reproduzir k seria
exatamente o ajuste empírico que o §8 do pedido veda.

---

## BINARY DIFFUSION

30 espécies ⇒ 30×29/2 = **435 pares**. Nenhum armazenado: todos calculados de σ, ε e M por
Chapman–Enskog com regras de Lorentz–Berthelot.

| Verificação | Resultado |
|---|---|
| Todos os 435 pares calculam | ✅ |
| D_ij = D_ji (tolerância 1e-14) | ✅ 435 de 435 |
| D_ij > 0 | ✅ 435 de 435 |
| D ∝ 1/P | ✅ exato |
| D cresce com T mais rápido que linearmente | ✅ |
| Limite binário de Blanc: D_A,m = D_AB | ✅ exato a 1e-12 |

Faixa: 2,05e-06 m²/s (CCl₄–I₂) a 1,61e-04 m²/s (H₂–He), razão 78×.

---

## MULTICOMPONENT REGRESSION

| Caso | n | Resultado |
|---|---:|---|
| Puro | 1 | ✅ |
| Binário | 2 | ✅ |
| Ternário | 3 | ✅ |
| Ar seco | 4 | ✅ |
| Ar úmido | 5 | ✅ |
| 15 componentes | 15 | ✅ |
| **30 componentes** | 30 | ✅ |
| Invariância à ordem das espécies | — | ✅ exato a 1e-14 |
| Nenhum limite `MAX_COMPONENTS` | — | ✅ |

**Wilke continua ativo**, com teste de regressão dedicado: He/Xe 50/50 dá μ_mix maior que a de
ambos os puros, o máximo não monotônico que só a regra de Wilke produz, e difere da média
ponderada em mais de 10 %.

---

## PERFORMANCE

### Profiling (§17, antes de otimizar)

Mistura de 30 componentes, 170 ms/ponto na medição inicial:

| Componente | Custo | Observação |
|---|---:|---|
| `anharmonic.py` (enumeração de níveis) | **67 %** | `term_value_cm1` chamado 117 700 vezes |
| `hindered_rotor.py` (Mathieu) | ~15 % | rediagonalizado a cada chamada |
| `binary_diffusion` | ~5 % | 900 chamadas por ponto |
| Wilke φ_ij | ~2 % | 900 avaliações |
| resto | ~11 % | |

Os dois primeiros eram **desperdício puro**: ambas as estruturas dependem só de constantes
imutáveis da espécie, mas eram reconstruídas a cada `Thermodynamics(...)`. A da variedade
anarmônica foi introduzida na tarefa anterior deste projeto.

### Correção 1 — memoização das estruturas invariantes

`functools.lru_cache` em `_level_temperatures` (chaveado pela `Anharmonicity`, congelada e
hashável) e em `torsional_levels_kelvin`. A escada torsional é devolvida como array somente-leitura
para que uma escrita acidental não envenene o cache.

| n | Antes | Depois | Ganho |
|---:|---:|---:|---:|
| 2 | 0,9 ms | 0,56 ms | 1,6× |
| 5 | 16,9 ms | 1,77 ms | 9,5× |
| 15 | 76,7 ms | 7,10 ms | 10,8× |
| 30 | 80,7 ms | 13,19 ms | 6,1× |

### Correção 2 — kernel numérico vetorizado

`statthermopy.transport.TransportKernel` — API de alto nível preservada intacta, kernel separado
para arrays (§18). Tudo que não depende do estado local vai para o construtor:

* arrays por espécie: M, σ, ε/k, δ;
* arrays por par: σ_ij², ε_ij, massas reduzidas e as metades puramente mássicas do fator de Wilke;
* tabela C_v,i(T), para que as somas quânticas não rodem no laço.

A chamada avalia arrays inteiros por *broadcasting*: `T`, `P` de forma `(...)` e `X` de forma
`(..., n_species)`. **Não lê YAML, não constrói `Molecule`, não indexa por string e não cria
objeto por célula.**

### Benchmarks (§22)

| n | pontos | referência [s] | kernel [s] | pontos/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 000 | 0,51 | 0,0007 | 1,39e+06 | 714× |
| 2 | 1 000 000 | 513 | 0,84 | 1,19e+06 | 609× |
| 5 | 1 000 | 1,78 | 0,0026 | 3,81e+05 | 679× |
| 5 | 100 000 | 178 | 0,32 | 3,09e+05 | 551× |
| 5 | 1 000 000 | 1 784 | 3,22 | 3,11e+05 | 554× |
| 15 | 100 000 | 636 | 2,45 | 4,08e+04 | 259× |
| 30 | 100 000 | 1 411 | 9,66 | 1,04e+04 | 146× |

Ganho acumulado sobre a medição original da auditoria (81 ms/ponto para 30 componentes):
**12 pontos/s → 10 400 pontos/s, ≈ 860×**. Para as 5 espécies típicas de CFD atmosférico:
**≈ 5 300×**.

### Equivalência física (§23)

O kernel reproduz o caminho de referência:

| Grandeza | Concordância |
|---|---|
| μ, ρ, M_mix, D_im | **exatos**, ≤ 1e-12 (precisão de máquina) |
| k, Pr, α, ν, c_p, γ | ≤ **2,4e-06** (interpolação da tabela de C_v) |

Verificado em 8 misturas, 5 temperaturas e 4 pressões, incluindo a mistura de 30 componentes.
Extrapolar fora da faixa tabulada é **recusado**, não silencioso.

### Memória

A matriz D_ij domina: `n_células × n² × 8 bytes`. Para 1 milhão de células, 5 espécies são
200 MB (viável); 30 espécies seriam 7,2 GB (inviável sem *chunking*). O kernel não faz *chunking*
automático — cabe ao chamador dividir o domínio, o que é natural em CFD.

---

## CFD READINESS

**PROPERTY-CALCULATOR READY: sim.** Sempre esteve; agora com condições-limite robustas e semântica
de Sc/Le correta.

**CFD-KERNEL READY: parcialmente.** O que está pronto:

* caminho vetorizado sem objetos por célula, validado contra a referência;
* pré-computação de todos os invariantes;
* 310 000 pontos/s com 5 espécies;
* *broadcasting* puro NumPy, portável para CuPy trocando o módulo de array;
* composição esparsa: uma espécie a zero cai fora de todas as somas.

O que falta para integração de produção:

* **chunking** de memória para n grande (o chamador precisa fazê-lo);
* **caminho CuPy/Numba** ainda não exercitado (a estrutura permite, mas não há teste de GPU);
* **`float32`** não suportado — tudo em dupla precisão;
* acurácia de k em espécies polares (§THERMAL CONDUCTIVITY) permanece o limite físico, não
  computacional.

Impacto nas equações, com os erros atuais:

| Equação | Coeficiente | Erro | Consequência |
|---|---|---:|---|
| Momentum ∇·[μ(∇u+∇uᵀ)] | μ_mix | ~1,7 % | Re dentro de 2 %; desprezível |
| Energia ∇·(k∇T) | k_mix | 3,9 % (não polar) / 15 % (polar) | **dominante**; afeta Nu |
| Espécies ∇·(ρD∇Y) | D_i,mix | ~5 % (Blanc) | moderado |

Para escoamento com vapor d'água, k(H₂O) segue sendo o item mais sério.

---

## REMAINING LIMITATIONS

1. **k de espécies polares** — 15,2 % médio nos quatro polares, 24,5 % em H₂O. Causa
   identificada (relaxação rotacional); correção exigiria Mason–Monchick com Z_rot.
2. **Stockmayer só em H₂O** — NH₃, SO₂ e H₂S têm o bloco disponível mas não recebem, porque a
   medição não mostrou ganho. Revisitar se surgirem parâmetros melhores.
3. **Difusão binária usa LJ mesmo para pares polares** — o refinamento de Stockmayer atinge só os
   coeficientes puros. Consistente e documentado, mas não é o tratamento de Brokaw completo.
4. **Blanc, não Maxwell–Stefan** — adequado para traço diluído; insuficiente se duas espécies não
   traço tiverem difusividades muito diferentes e gradientes opostos.
5. **Sem correções de gás denso** — Tc/Pc/ω estão armazenados como gancho, nenhum caminho os
   consome.
6. **Kernel sem chunking nem GPU exercitada.**
7. **`test_gui.py`** continua travando em `test_theme_toggle_via_menu`, defeito pré-existente
   alheio a este trabalho.

---

## ARQUIVOS ALTERADOS

| Arquivo | Mudança |
|---|---|
| `database/data/NO.yaml` | aspas em `name`/`formula` |
| `validation/data/NO.yaml` | aspas em `species` |
| `database/data/H2O.yaml` | bloco `stockmayer` |
| `core/molecule.py` | dataclass `Stockmayer`; campo em `Molecule` |
| `database/registry.py` | leitura do bloco `stockmayer` |
| `transport/collision.py` | `omega_11`/`omega_22` aceitam δ (Brokaw) |
| `transport/transport.py` | `_potential()` escolhe o conjunto da espécie |
| `transport/kernel.py` | **novo** — kernel vetorizado |
| `transport/__init__.py` | exporta `TransportKernel` |
| `transport/air/mixture_transport.py` | `trace=None`; `D_im`/`Sc_i`/`Le_i`; acessores |
| `transport/air/air_transport.py` | declara `trace="H2O"` |
| `mixture.py` | frações zero e negativas |
| `modes/anharmonic.py` | cache da variedade de níveis |
| `modes/hindered_rotor.py` | cache da escada torsional |
| `tests/test_database.py` | 4 testes de integridade YAML |
| `tests/test_transport_audit.py` | **novo** — 47 testes |
| `tests/test_transport_kernel.py` | **novo** — 38 testes |
| `tests/test_air_transport.py` | traçador explícito no caminho de referência |

**646 testes**, todos passando (exceto o `test_gui.py` pré-existente).
