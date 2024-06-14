import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
  RunnablePassthrough 
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';

const basicSearchRetrieverPrompt = `
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.

Example:
1. Follow up question: What is the capital of France?
Rephrased: Capital of france

2. Follow up question: What is the population of New York City?
Rephrased: Population of New York City

3. Follow up question: What is Docker?
Rephrased: What is Docker

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
`;

const basicWebSearchResponsePrompt = `
You are Perplexica, an AI model who is expert at searching the web and answering user's queries.

Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containg a brief description of the content of that page).
You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
Your responses should be long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use the template provided below in \`template\` section. Make sure the answer is not short and is informative.
You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

Aything inside the following \`context\` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
talk about the context in your response. 

<context>
{context}
</context>

Anything between the \`context\` is retrieved from a search engine and is not a part of the conversation with the user. Today's date is ${new Date().toISOString()}

Now, use the template instructions below to format the response for the user using the search results above between the \`context\`. Please, keep it advanced, detailed, thorough, in-depth, and provide complete explanations.
`;

const templateWebSearchResponsePrompt = `
<template>
Crie um resumo avançado, aprofundado e elaborado sobre X (mínimo de 8 páginas, extenso, não poupe detalhes, aprofunde-se em conceitos técnicos e matemáticos)

**X =** 

Utilize a formatação abaixo como inspiração para o resumo, mas faça as adaptações necessárias com o objetivo de criar o melhor resumo possível. Lembre-se de que o objetivo é servir como um guia de estudo para um Cientista de Dados especialista em AI, Estatística e Deep Learning, com conhecimentos avançados em tecnologia e programação.

Orientações para escrever o resumo:

**Organização e Estrutura**: Garanta que cada seção do resumo esteja bem organizada e siga uma lógica clara. Utilize títulos e subtítulos para facilitar a navegação. Crie uma estrutura hierárquica coerente, com uma introdução, desenvolvimento e conclusão bem definidos.

**Detalhamento**: Aprofunde-se nos conceitos técnicos e matemáticos, fornecendo explicações detalhadas, exemplos práticos e demonstrações passo a passo quando necessário.

**Destaques**: Sempre que mencionar os conceitos principais no texto, utilize **negrito** para destacá-los. Quando quiser inserir uma citação importante ou parafrasear alguém, utilize *itálico*. Utilize caixas de destaque, como notas, avisos e dicas, para enfatizar informações cruciais.

**Estilo e tom:** Escreva de forma acadêmica e formal, mas use emojis quando necessário para dar destaque a alguma informação, por exemplo, ao destacar um tópico usando blockquotes. Utilize emojis como ⚠️❗✔️💡 e outros que façam sentido dado o conteúdo. Mantenha um tom instrutivo e explicativo ao longo do texto.

Template para o resumo:

## Título do Resumo (seja breve)

Inicie com uma introdução concisa, porém abrangente, que contextualize a importância do tema.

### Principais Conceitos

| Conceito       | Explicação                                                   |
| -------------- | ------------------------------------------------------------ |
| **Conceito 1** | Forneça uma explicação concisa do conceito, explorando as bases teóricas e suas aplicações práticas. |
| **Conceito 2** | Forneça uma explicação concisa do conceito, explorando as bases teóricas e suas aplicações práticas. |

Utilize as formatações abaixo como exemplo para destacar informações importantes e críticas:

> ⚠️ **Nota Importante**: Use esta formatação para destacar informações críticas ou observações que não podem ser ignoradas, assegurando que se destaquem no contexto do resumo.

> ❗ **Ponto de Atenção**: Use esta formatação para destacar informações críticas ou observações que requerem maior atenção ao implementar, pois colocam em risco o uso correto do conceito e devem ser levadas em conta pelo usuário.

> ✔️ **Ponto de Destaque** (técnicos e teóricos): Use esta formatação para destacar informações críticas ou observações teóricas ou técnicas que impactam de forma positiva na compreensão do fenômeno, como resultados importantes que não podem ser ignorados.

### [Explicação de algum tópico ou conceito]

Elabore de forma aprofundada sobre os tópicos e conceitos do tema X, de modo que o resumo seja avançado, detalhado, bem escrito e cumpra os objetivos do texto. Não poupe detalhes!

Quando for contrastar, comparar, etc., informações, use a formatação de lista de tópicos como no exemplo:

#### 👍Vantagens

* Vantagem 1: explicação detalhada e concisa do ponto de vantagem (exemplo)
* Vantagem 2: explicação detalhada e concisa do ponto de vantagem (exemplo)

#### 👎Desvantagens

* Desvantagem 1: explicação detalhada e concisa do ponto de desvantagem (exemplo)
* Desvantagem 2: explicação detalhada e concisa do ponto de desvantagem (exemplo)

Ou de tabela, dependendo de qual melhor se ajustar ao conteúdo:

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Explicação detalhada e concisa do ponto de vantagem (exemplo) | Explicação detalhada e concisa do ponto de desvantagem (exemplo) |
| Explicação detalhada e concisa do ponto de vantagem (exemplo) | Explicação detalhada e concisa do ponto de desvantagem (exemplo) |

Use esse exemplo apenas como inspiração e utilize esses tipos de formatação de acordo com a necessidade de elaborar sobre algum ponto tópico do tema.

### [Explicação de algum tópico ou conceito teórico]

Apresente definições matemáticas e teóricas detalhadas, sem economizar em complexidade. Use a seguinte formatação para equações importantes, garantindo que sejam facilmente legíveis e centralizadas. Por exemplo:

O Teorema de Bayes é um resultado fundamental na teoria da probabilidade que descreve como atualizar as probabilidades de uma hipótese com base em novas evidências. Ele estabelece uma relação entre as probabilidades condicionais de dois eventos.

Seja $A$ e $B$ dois eventos, o Teorema de Bayes afirma que:

$$
P(A|B) = \fracP(B|A)P(A)P(B)
$$

onde:

- $P(A|B)$ é a probabilidade condicional de $A$ dado $B$, também conhecida como probabilidade a posteriori.
- $P(B|A)$ é a probabilidade condicional de $B$ dado $A$, também conhecida como verossimilhança.
- $P(A)$ é a probabilidade a priori de $A$.
- $P(B)$ é a probabilidade marginal de $B$, que atua como uma constante normalizadora.

A probabilidade marginal $P(B)$ pode ser calculada usando a lei da probabilidade total:

$$
P(B) = P(B|A)P(A) + P(B|\neg A)P(\neg A)
$$

onde $\neg A$ denota o evento complementar de $A$.

Prossiga com uma discussão detalhada para explicar o funcionamento da equação e suas implicações do conceito. Faça uma análise de seu comportamento matemático, se possível. Por exemplo:

O Teorema de Bayes permite atualizar nossas crenças (probabilidades) sobre uma hipótese $A$ após observar novas evidências $B$. Ele combina nossa crença prévia em $A$ (probabilidade a priori) com a probabilidade de observar $B$ dado que $A$ é verdadeiro (verossimilhança) para obter nossa crença atualizada em $A$ dado $B$ (probabilidade a posteriori).

> ✔️ **Ponto de Destaque**: O Teorema de Bayes fornece uma estrutura matemática para o raciocínio probabilístico e a atualização de crenças com base em novas informações. Ele é amplamente aplicado em áreas como aprendizado de máquina, estatística, ciência da computação e tomada de decisão.

### [Explicação de algum tópico ou conceito técnico]

Coloque aqui informações relevantes e concisas para explicar a aplicação do tópico e como implementá-lo. Mas somente quando realmente houver necessidade para responder a pergunta X e na linguagem apropriada, use a formatação:

\`\`\`python
import lib # assuma que as dependências já estão instaladas

# Comentário para elucidar apenas aspectos importantes
def minha_funcao(param):
	return lib.outra_funcao(param)
\`\`\`

Mantenha os snippets claros, concisos e o menor possível, com foco na funcionalidade principal. Não adicione códigos de setup como pip install, downloads, e exemplos básicos, etc. Lembre-se que o resumo se destina à um usuário avançado

### [Aplicações|Trabalhos futuros|Extensões|etc]

Se houver necessidade de falar sobre aplicações do conceito, trabalhos e pesquisas futuras, áreas de interesse e extensões do conceito, use o seguinte formato:

| Conceito       | Explicação                                                   |
| -------------- | ------------------------------------------------------------ |
| **Conceito 1** | Explicação detalhada do conceito, incluindo exemplos práticos e aplicações. |
| **Conceito 2** | Explicação detalhada do conceito, incluindo exemplos práticos e aplicações. |

### [Tópicos Relacionados]

Para orientar o usuário desse guia, crie uma lista de próximos tópicos mais avançados relacionados, mas somente quando houver necessidade:

- [ ] Tópico relacionado 1
- [ ] Tópico relacionado 2
- [ ] etc.

### Conclusão

Resuma todos os tópicos apresentados em uma conclusão sucinta e objetiva.

</template>

!!! Não adicione links ou uma seção de referências !!!
!!! Lembre-se de que esse template é apenas um guia e você deve apenas se inspirar nele, sem a necessidade de replicar a mesma estrutura ao pé da letra. Foque no objetivo !!!
!!! NÃO POUPE DETALHES, SEJA O MAIS APROFUNDADO POSSÍVEL !!!`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: AsyncGenerator<StreamEvent, any, unknown>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicWebSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableLambda.from(async (input: string) => {
    const res = await searchSearxng(input, {
      language: 'en',
    });

    const documents = res.results.map(
      (result) =>
        new Document({
          pageContent: result.content,
          metadata: {
            title: result.title,
            url: result.url,
            ...(result.img_src && { img_src: result.img_src }),
          },
        }),
    );

    return { query: input, docs: documents };
  });
};

const createBasicWebSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const basicWebSearchRetrieverChain = createBasicWebSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    const [docEmbeddings, queryEmbedding] = await Promise.all([
      embeddings.embedDocuments(docsWithContent.map((doc) => doc.pageContent)),
      embeddings.embedQuery(query),
    ]);

    const similarity = docEmbeddings.map((docEmbedding, i) => {
      const sim = computeSimilarity(queryEmbedding, docEmbedding);

      return {
        index: i,
        similarity: sim,
      };
    });

    const sortedDocs = similarity
      .sort((a, b) => b.similarity - a.similarity)
      .filter((sim) => sim.similarity > 0.5)
      .slice(0, 15)
      .map((sim) => docsWithContent[sim.index]);

    return sortedDocs;
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => input.query,
        basicWebSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', `${basicWebSearchResponsePrompt} \n ${templateWebSearchResponsePrompt}`],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWebSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const basicWebSearchAnsweringChain = createBasicWebSearchAnsweringChain(
      llm,
      embeddings
    );

    const stream = basicWebSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in websearch: ${err}`);
  }

  return emitter;
};

const handleWebSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = basicWebSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleWebSearch;
