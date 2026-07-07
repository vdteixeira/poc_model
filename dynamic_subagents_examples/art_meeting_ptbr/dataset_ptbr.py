"""Dataset SINTÉTICO em pt-BR — análogo do dataset de e-mails do ART-E.

No ART-E os "cenários" são perguntas sobre uma caixa de e-mail; aqui são
perguntas sobre transcrições de reunião. O ponto que resolve o seu "não
tenho dados de português": **o RULER não precisa de rótulos** — ele julga
as tentativas relativamente. Você só precisa de CENÁRIOS (transcrição +
pergunta). Este arquivo gera esses cenários à mão, em português.

Cada reunião é dividida em SEGMENTOS (falante + texto), imitando a saída
real do seu pipeline de STT+diarização. O agente vai buscar e ler
segmentos para responder — é o análogo direto do `search_inbox`/`read_email`.

O campo `expected` é OPCIONAL e serve só para métricas de validação
(um "juiz de correção", como no ART-E). O treino em si usa RULER.
"""

from dataclasses import dataclass, field


@dataclass
class Segment:
    idx: int
    speaker: str
    text: str


@dataclass
class MeetingScenario:
    id: str
    title: str
    date: str
    segments: list[Segment]
    question: str
    # Só para validação/juiz de correção; RULER não usa.
    expected: str = ""
    split: str = "train"  # "train" | "val"


def _seg(rows: list[tuple[str, str]]) -> list[Segment]:
    return [Segment(i, sp, tx) for i, (sp, tx) in enumerate(rows)]


SCENARIOS: list[MeetingScenario] = [
    MeetingScenario(
        id="planning-q3",
        title="Planejamento Q3 — Produto",
        date="2026-06-10",
        segments=_seg([
            ("Ana", "Bom dia pessoal. Hoje precisamos fechar o escopo do Q3."),
            ("Bruno", "A prioridade número um continua sendo o app de desktop, certo?"),
            ("Ana", "Sim. O desktop tem que sair até o fim de julho, é compromisso com o cliente."),
            ("Carla", "Eu assumo a parte de captura de áudio do desktop então."),
            ("Bruno", "E eu fico com a integração do Teams. Fecho a POC até dia 20."),
            ("Ana", "Perfeito. A Carla entrega captura até 25 de julho e o Bruno a POC do Teams até 20 de junho."),
            ("Carla", "Combinado. Alguma dependência externa que eu precise saber?"),
            ("Ana", "Só o SDK de captura, mas isso já está aprovado pelo jurídico."),
        ]),
        question="Quais são as tarefas atribuídas e seus responsáveis e prazos?",
        expected=(
            "Carla: captura de áudio do desktop até 25/07. "
            "Bruno: POC de integração do Teams até 20/06."
        ),
        split="train",
    ),
    MeetingScenario(
        id="incident-review",
        title="Revisão de Incidente — Fila de Transcrição",
        date="2026-06-12",
        segments=_seg([
            ("Diego", "A fila de transcrição travou ontem às 14h por duas horas."),
            ("Elena", "A causa foi o pico de reuniões simultâneas, o worker não escalou."),
            ("Diego", "Precisamos de auto-scaling na fila até o fim da semana."),
            ("Elena", "Eu configuro o auto-scaling. Sexta no máximo."),
            ("Diego", "E vamos adicionar um alerta de profundidade de fila também."),
            ("Fábio", "Eu crio o alerta de fila, posso entregar na quinta."),
            ("Diego", "Ótimo. Nenhum dado de cliente foi perdido, só houve atraso."),
        ]),
        question="O que causou o incidente e quais ações foram definidas para evitar recorrência?",
        expected=(
            "Causa: pico de reuniões simultâneas sem auto-scaling do worker. "
            "Ações: Elena configura auto-scaling até sexta; Fábio cria alerta de "
            "profundidade de fila até quinta."
        ),
        split="train",
    ),
    MeetingScenario(
        id="sales-sync",
        title="Sync Comercial — Conta Northwind",
        date="2026-06-15",
        segments=_seg([
            ("Gabriela", "A Northwind quer expandir de 50 para 300 licenças."),
            ("Hugo", "Ótimo. Eles pediram algum requisito novo?"),
            ("Gabriela", "Sim, isolamento de dados por região e retenção de 90 dias."),
            ("Hugo", "Retenção configurável já temos. Isolamento por região é roadmap."),
            ("Gabriela", "Eles topam esperar o isolamento se tivermos data marcada."),
            ("Hugo", "Vou levantar o esforço de isolamento por região com o time de dados."),
            ("Gabriela", "Fecho a proposta de 300 licenças até quarta que vem."),
        ]),
        question="Quais requisitos o cliente pediu e qual é o próximo passo de cada pessoa?",
        expected=(
            "Requisitos: isolamento de dados por região e retenção de 90 dias. "
            "Próximos passos: Hugo levanta o esforço de isolamento por região; "
            "Gabriela fecha a proposta de 300 licenças até quarta."
        ),
        split="train",
    ),
    MeetingScenario(
        id="design-review",
        title="Design Review — Bot do Teams",
        date="2026-06-18",
        segments=_seg([
            ("Iara", "O @TagBot precisa responder menção em canal e em DM."),
            ("João", "Em canal, só responde se for explicitamente mencionado, senão vira spam."),
            ("Iara", "Concordo. E em DM responde sempre."),
            ("João", "A latência alvo é abaixo de 3 segundos para a primeira resposta."),
            ("Iara", "Eu escrevo a spec da máquina de estados do bot até segunda."),
            ("João", "Eu prototipo o handler de menção até quarta."),
        ]),
        question="Quais decisões de comportamento do bot foram tomadas e quem entrega o quê?",
        expected=(
            "Decisões: responde em canal só quando mencionado, sempre em DM, "
            "latência alvo <3s. Iara escreve a spec até segunda; João prototipa "
            "o handler de menção até quarta."
        ),
        split="train",
    ),
    MeetingScenario(
        id="retention-policy",
        title="Política de Retenção e Consentimento",
        date="2026-06-20",
        segments=_seg([
            ("Lara", "Todo áudio gravado precisa de consentimento explícito do organizador."),
            ("Marcos", "E a retenção padrão do áudio bruto?"),
            ("Lara", "Trinta dias para o áudio, um ano para a transcrição."),
            ("Marcos", "Cada tenant pode encurtar, mas não estender além disso."),
            ("Lara", "Exato. Eu documento a política até sexta."),
            ("Marcos", "Eu implemento o job de expiração de áudio até o fim do mês."),
        ]),
        question="Quais são as regras de consentimento e retenção definidas?",
        expected=(
            "Consentimento explícito do organizador para gravar áudio. Retenção: "
            "30 dias para áudio bruto, 1 ano para transcrição; tenant pode "
            "encurtar mas não estender. Lara documenta até sexta; Marcos "
            "implementa o job de expiração até o fim do mês."
        ),
        split="train",
    ),
    MeetingScenario(
        id="hiring-plan",
        title="Plano de Contratação — Time de IA",
        date="2026-06-22",
        segments=_seg([
            ("Núbia", "Precisamos de dois engenheiros de ML no Q3."),
            ("Otávio", "Um sênior para o pipeline de RL e um pleno para avaliação."),
            ("Núbia", "Eu abro as duas vagas até quinta."),
            ("Otávio", "Eu preparo o desafio técnico de RL até a semana que vem."),
            ("Núbia", "Meta é fechar as contratações até o fim de agosto."),
        ]),
        question="Quantas vagas, com quais perfis, e quais são as ações e prazos?",
        expected=(
            "Duas vagas: um ML sênior (pipeline de RL) e um pleno (avaliação). "
            "Núbia abre as vagas até quinta; Otávio prepara o desafio técnico "
            "até a semana seguinte; meta de fechar até fim de agosto."
        ),
        split="val",
    ),
    MeetingScenario(
        id="pricing-debate",
        title="Debate de Precificação",
        date="2026-06-24",
        segments=_seg([
            ("Paula", "O plano por assento não está funcionando para clientes grandes."),
            ("Rui", "Proponho cobrança híbrida: assento mais consumo de minutos transcritos."),
            ("Paula", "Risco de assustar o cliente com conta variável."),
            ("Rui", "Colocamos um teto mensal para dar previsibilidade."),
            ("Paula", "Gostei. Eu modelo três cenários de receita até terça."),
            ("Rui", "Eu levanto o custo real de transcrição por minuto até segunda."),
        ]),
        question="Qual mudança de precificação foi proposta e quais tarefas ficaram definidas?",
        expected=(
            "Proposta: cobrança híbrida (assento + consumo de minutos "
            "transcritos) com teto mensal. Paula modela três cenários de receita "
            "até terça; Rui levanta o custo real por minuto até segunda."
        ),
        split="val",
    ),
    MeetingScenario(
        id="empty-decisions",
        title="Bate-papo sem decisões",
        date="2026-06-25",
        segments=_seg([
            ("Sofia", "Só queria alinhar como todos estão, sem pauta fechada hoje."),
            ("Tiago", "Por mim tudo tranquilo, semana calma."),
            ("Sofia", "Ótimo. Então não temos nada a decidir, bom descanso a todos."),
        ]),
        question="Quais tarefas e responsáveis foram definidos nesta reunião?",
        # Cenário de controle: a resposta correta é "nenhuma". Ensina o
        # modelo a NÃO inventar tarefas — o análogo do "arquivo limpo".
        expected="Nenhuma tarefa foi definida; a reunião não teve decisões.",
        split="val",
    ),
]


def train_scenarios() -> list[MeetingScenario]:
    return [s for s in SCENARIOS if s.split == "train"]


def val_scenarios() -> list[MeetingScenario]:
    return [s for s in SCENARIOS if s.split == "val"]
