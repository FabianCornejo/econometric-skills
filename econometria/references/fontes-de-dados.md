# Fontes de dados para econometria

Prioridade: (1) arquivos que o usuário anexar; (2) APIs oficiais via Python; (3) conectores/plugins habilitados; (4) busca na web para localizar uma série específica. Registre sempre a fonte, o código da série, a data de extração e a unidade.

## Brasil

### Banco Central — SGS (Sistema Gerenciador de Séries Temporais)

API pública, sem chave. Formato:

```
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{CODIGO}/dados?formato=json&dataInicial=01/01/2015&dataFinal=31/12/2025
https://api.bcb.gov.br/dados/serie/bcdata.sgs.{CODIGO}/dados/ultimos/{N}?formato=json
```

Séries diárias têm limite de 10 anos por consulta: quebre em janelas.

```python
import pandas as pd, requests

def sgs(codigo, inicio="01/01/2010", fim=None):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial={inicio}"
    if fim:
        url += f"&dataFinal={fim}"
    df = pd.DataFrame(requests.get(url, timeout=60).json())
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.set_index("data")["valor"].rename(codigo)
```

Alternativa: `pip install --break-system-packages python-bcb` → `from bcb import sgs; sgs.get({"selic": 432}, start="2015-01-01")` (lida com a quebra em janelas sozinho).

Códigos úteis (confirme o nome no portal do SGS antes de publicar):

| Código | Série | Frequência |
|---|---|---|
| 432 | Meta Selic definida pelo Copom (% a.a.) | diária |
| 11 | Selic diária (% a.d.) | diária |
| 4390 | Selic acumulada no mês (% a.m.) | mensal |
| 433 | IPCA — variação mensal (%) | mensal |
| 13522 | IPCA — acumulado 12 meses (%) | mensal |
| 189 | IGP-M — variação mensal (%) | mensal |
| 1 | Dólar comercial PTAX venda (R$/US$) | diária |
| 3698 | Dólar PTAX venda — média do mês | mensal |
| 24363 | IBC-Br (sem ajuste sazonal) | mensal |
| 24364 | IBC-Br (com ajuste sazonal) | mensal |
| 4380 | PIB mensal — valores correntes (R$ milhões) | mensal |
| 24369 | Taxa de desocupação — PNAD Contínua (%) | mensal (trimestre móvel) |

Expectativas do mercado (Focus): API Olinda `https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/` ou `from bcb import Expectativas`.

### IBGE — SIDRA

```
https://apisidra.ibge.gov.br/values/t/{TABELA}/n1/all/v/{VARIAVEL}/p/{PERIODOS}
```

- `n1/all` = Brasil; `n3/all` = UFs; `n6/{código}` = município.
- `p/all`, `p/last 12`, `p/202001-202512`.
- A primeira linha da resposta é o cabeçalho: descarte-a.
- Exemplo: IPCA variação mensal → tabela 1737, variável 63.
- Para descobrir tabela/variável: https://sidra.ibge.gov.br (pesquisa por tema) ou `https://servicodados.ibge.gov.br/api/v3/agregados/{TABELA}/metadados`.
- Pacote: `sidrapy` (`sidrapy.get_table(table_code="1737", territorial_level="1", ibge_territorial_code="all", variable="63", period="last 24")`).

Pesquisas mais usadas: IPCA/INPC, PMC (comércio), PMS (serviços), PIM-PF (indústria), PNAD Contínua, Contas Nacionais Trimestrais (PIB).

### Ipeadata

API OData sem chave; agrega séries de várias fontes (BCB, IBGE, FGV, Funcex...) com séries longas.

```
http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='BM12_TJOVER12')
http://www.ipeadata.gov.br/api/odata4/Metadados('BM12_TJOVER12')
```

Resposta em `json["value"]`, campos `VALDATA` e `VALVALOR`. Pacote: `ipeadatapy`.

### Outras fontes brasileiras
- CVM (dados abertos de companhias abertas): https://dados.cvm.gov.br
- Tesouro Nacional (títulos públicos, curva): Tesouro Transparente.
- B3 (preços, índices): sem API pública ampla; preferir arquivos que o usuário fornecer.
- Receita Federal (arrecadação): dados abertos.

## Internacionais

| Fonte | Como acessar | Bom para |
|---|---|---|
| FRED (Fed St. Louis) | `pandas_datareader` ou API com chave gratuita | Séries dos EUA e muitas globais |
| Banco Mundial | API `https://api.worldbank.org/v2/country/BRA/indicator/{IND}?format=json` ou `wbgapi` | Painéis de países |
| FMI | API SDMX; pacote `imfp` | WEO, IFS, balanço de pagamentos |
| OCDE | API SDMX `https://sdmx.oecd.org/public/rest/` | Indicadores comparáveis |
| Eurostat | pacote `eurostat` | Europa |

Se o plugin **Monitly** estiver habilitado na conversa, ele dá acesso direto a OCDE, Banco Mundial, FMI, Eurostat e institutos nacionais. Se o conector **Oxford Economics** estiver conectado (exige assinatura), use-o para projeções e dados macro proprietários. Verifique se estão ativos antes de usar; não presuma.

## Se o acesso à rede falhar

O ambiente de trabalho às vezes bloqueia domínios externos. Se uma chamada falhar:
1. Não tente contornar com outro comando — o bloqueio é o mesmo.
2. Peça ao usuário para exportar o CSV/XLSX do portal (SGS, SIDRA e Ipeadata têm botão de exportação) e anexar.
3. Para poucas observações, a busca na web pode localizar valores, mas confirme na fonte oficial.

## Boas práticas de dados
- Documente unidades: % a.m. vs. % a.a., índice vs. variação, R$ correntes vs. constantes.
- Converta taxas mensais em índices encadeados antes de deflacionar: `indice = (1 + ipca/100).cumprod()`.
- Alinhe datas ao início do mês (`to_period("M")`) para unir séries mensais de fontes diferentes.
- Prefira séries sem ajuste sazonal + tratamento próprio quando for modelar sazonalidade; use as dessazonalizadas oficiais quando o foco for outra coisa.
- Guarde os dados brutos extraídos num CSV junto com o script, para reprodutibilidade.
