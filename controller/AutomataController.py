import json
from tkinter import Tk, filedialog
import tkinter
import networkx as nx
import matplotlib.pyplot as plt
import tkinter
from tkinter import ttk
from PIL import Image, ImageTk
import tkinter as tk


def epsilon_closure(states, transitions, epsilon):
    """
    La función calcula el cierre épsilon de un conjunto de estados en un conjunto dado de transiciones.

    :param states: Un conjunto de estados en el autómata finito
    :param transitions: El parámetro transiciones es un diccionario que representa las transiciones
    entre estados en un autómata finito. Las claves del diccionario son los estados y los valores son
    diccionarios que representan las transiciones desde ese estado. Las claves de los diccionarios
    internos son los símbolos que se pueden leer desde la entrada y los valores
    :param epsilon: epsilon es un símbolo utilizado en la teoría de los autómatas para representar una
    transición vacía. En otras palabras, representa una transición que se puede realizar sin consumir
    ningún símbolo de entrada. La función epsilon_closure toma un conjunto de estados, un diccionario de
    transiciones y el símbolo epsilon como entrada y devuelve el cierre epsilon
    :return: un conjunto congelado de estados que se puede alcanzar desde el conjunto de estados de
    entrada siguiendo las transiciones épsilon en un conjunto dado de transiciones.
    """
    epsilon_closure_set = set(states)
    stack = list(states)

    while stack:
        state = stack.pop()
        if state in transitions and epsilon in transitions[state]:
            epsilon_states = transitions[state][epsilon]
            for epsilon_state in epsilon_states:
                if epsilon_state not in epsilon_closure_set:
                    epsilon_closure_set.add(epsilon_state)
                    stack.append(epsilon_state)

    return frozenset(epsilon_closure_set)


def move(states, transitions, symbol):
    """
    La función toma un conjunto de estados, un diccionario de transiciones y un símbolo, y devuelve un
    conjunto congelado de estados que se pueden alcanzar desde los estados de entrada usando el símbolo
    dado.

    :param states: Un conjunto de estados en el autómata
    :param transitions: El parámetro transiciones es un diccionario que representa las transiciones de
    un autómata finito. Las claves del diccionario son los estados del autómata, y los valores también
    son diccionarios. Los diccionarios internos representan las transiciones desde el estado
    correspondiente, donde las claves son símbolos y los valores son conjuntos de estados que
    :param symbol: El símbolo de entrada para el que queremos encontrar el conjunto de estados que se
    pueden alcanzar desde el conjunto de estados dado a través de las transiciones dadas
    :return: un conjunto congelado de estados al que se puede llegar desde el conjunto de estados de
    entrada utilizando el símbolo de entrada y las transiciones.
    """
    move_set = set()

    for state in states:
        if state in transitions and symbol in transitions[state]:
            move_states = transitions[state][symbol]
            move_set.update(move_states)

    return frozenset(move_set)


def load_nfa_from_json():
    """
    Esta función carga un NFA desde un archivo JSON, lo convierte en un DFA, visualiza el DFA y devuelve
    los estados, las transiciones, el estado inicial, los estados finales y el alfabeto del NFA.
    :return: una tupla que contiene los siguientes elementos:
    - nfa_states: un conjunto de estados en la NFA
    - nfa_transitions: un diccionario que representa las transiciones en el NFA
    - nfa_start_state: el estado de inicio de la NFA
    - nfa_final_states: un conjunto de estados finales en la NFA
    - alfabeto: un conjunto de símbolos en el
    """
    file_path = filedialog.askopenfilename(
        filetypes=[('JSON Files', '*.json')])

    if file_path:

        with open(file_path, 'r') as file:
            data = json.load(file)

    nfa_states = set(data['states'])
    nfa_transitions = data['transitions']
    nfa_start_state = data['start_state']
    nfa_final_states = set(data['final_states'])
    alphabet = set(data['alphabet'])

    # Conversión del NFA a DFA
    dfa = nfa_to_dfa((nfa_states, nfa_transitions,
                     nfa_start_state, nfa_final_states), alphabet)

    # Visualización del DFA
    visualize_dfa(dfa)
    show_nfa(nfa_states, nfa_transitions, nfa_start_state, nfa_final_states)

    return nfa_states, nfa_transitions, nfa_start_state, nfa_final_states, alphabet


def load_nfa_from_txt():
    """
    Esta función carga un NFA desde un archivo de texto, lo convierte en un DFA, visualiza el DFA y devuelve los estados,
    las transiciones, el estado inicial, los estados finales y el alfabeto del NFA.
    :return: una tupla que contiene los siguientes elementos:
    - nfa_states: un conjunto de estados en la NFA
    - nfa_transitions: un diccionario que representa las transiciones en el NFA
    - nfa_start_state: el estado de inicio de la NFA
    - nfa_final_states: un conjunto de estados finales en la NFA
    - alphabet: un conjunto de símbolos en el alfabeto
    """
    file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Procesar las líneas del archivo de texto
    nfa_states = set()
    nfa_transitions = {}
    nfa_start_state = ''
    nfa_final_states = set()
    alphabet = set()

    for line in lines:
        line = line.strip()

        if line.startswith('states:'):
            nfa_states = set(line.split(':')[1].split())
        elif line.startswith('transitions:'):
            transitions = {}
            for transition_line in lines[lines.index(line) + 1:]:
                transition_line = transition_line.strip()
                if not transition_line:
                    break

                transition_parts = transition_line.split()
                state = transition_parts[0]
                symbol = transition_parts[1]
                target_states = transition_parts[2:]

                if state not in transitions:
                    transitions[state] = {}

                transitions[state][symbol] = target_states

            nfa_transitions = transitions
        elif line.startswith('start_state:'):
            nfa_start_state = line.split(':')[1].strip()
        elif line.startswith('final_states:'):
            nfa_final_states = set(line.split(':')[1].split())
        elif line.startswith('alphabet:'):
            alphabet = set(line.split(':')[1].split())

    # Conversión del NFA a DFA
    dfa = nfa_to_dfa((nfa_states, nfa_transitions,
                     nfa_start_state, nfa_final_states), alphabet)

    # Visualización del DFA
    visualize_dfa(dfa)
    show_nfa(nfa_states, nfa_transitions, nfa_start_state, nfa_final_states)

    return nfa_states, nfa_transitions, nfa_start_state, nfa_final_states, alphabet



def visualize_dfa(dfa):
    """
    La función visualiza un autómata finito determinista (DFA) utilizando las bibliotecas networkx y
    matplotlib en Python.

    :param dfa: una tupla que contiene la información de DFA, incluido el conjunto de estados,
    transiciones, estado de inicio y estados finales
    """
    dfa_states, dfa_transitions, dfa_start_state, dfa_final_states = dfa

    graph = nx.DiGraph()

    for state in dfa_states:
        if state in dfa_final_states:
            graph.add_node(state, final=True)
        else:
            graph.add_node(state)

    for state, transitions in dfa_transitions.items():
        for symbol, target_state in transitions.items():
            graph.add_edge(state, target_state, label=symbol)

    pos = nx.spring_layout(graph)

    fig, ax = plt.subplots()

    nx.draw_networkx(
        graph,
        pos,
        node_color='lightblue',
        node_size=500,
        font_size=12,
        ax=ax
    )

    node_labels = {state: state for state in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax)

    edge_labels = {(u, v): data['label']
                   for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)

    ax.set_axis_off()

    # Obtén la ventana raíz de tkinter
    root = plt.gcf().canvas.manager.window
    root.title('Automata Determinista')

    plt.show()


def nfa_to_dfa(nfa, alphabet):
    """
    Esta función convierte un NFA dado (autómata finito no determinista) en un DFA (autómata finito
    determinista).

    :param nfa: una tupla que representa el NFA, que contiene el conjunto de estados, transiciones,
    estado de inicio y estados finales
    :param alphabet: El conjunto de símbolos que componen el alfabeto de entrada para NFA y DFA
    :return: La función `nfa_to_dfa` devuelve una tupla que contiene cuatro elementos:
    - `dfa_states`: una lista de estados en el DFA resultante
    - `dfa_transitions`: un diccionario que representa la función de transición del DFA resultante
    - `dfa_start_state`: el estado inicial del DFA resultante
    - `dfa_final_states`: una lista de estados finales en el DFA resultante
    """
    nfa_states, nfa_transitions, nfa_start_state, nfa_final_states = nfa

    dfa_states = []
    dfa_transitions = {}
    dfa_start_state = epsilon_closure(
        {nfa_start_state}, nfa_transitions, 'epsilon')
    dfa_final_states = []

    stack = [dfa_start_state]
    dfa_states.append(dfa_start_state)

    while stack:
        current_states = stack.pop()

        for symbol in alphabet:
            move_states = move(current_states, nfa_transitions, symbol)
            epsilon_closure_states = epsilon_closure(
                move_states, nfa_transitions, 'epsilon')

            if epsilon_closure_states not in dfa_states:
                stack.append(epsilon_closure_states)
                dfa_states.append(epsilon_closure_states)

            if current_states not in dfa_transitions:
                dfa_transitions[current_states] = {}

            dfa_transitions[current_states][symbol] = epsilon_closure_states

    for state in dfa_states:
        if nfa_final_states.intersection(state):
            dfa_final_states.append(state)

    return dfa_states, dfa_transitions, dfa_start_state, dfa_final_states


def show_nfa(nfa_states, nfa_transitions, nfa_start_state, nfa_final_states):
    """
    La función `show_nfa` toma un NFA y lo muestra como un gráfico utilizando las bibliotecas networkx y
    matplotlib en Python.

    :param nfa_states: Un conjunto de todos los estados en la NFA
    :param nfa_transitions: Un diccionario que representa las transiciones de la NFA. Las claves son los
    estados de origen y los valores son diccionarios donde las claves son los símbolos que se pueden
    leer desde el estado de origen y los valores son listas de estados de destino que se pueden alcanzar
    leyendo el símbolo correspondiente del estado de origen
    :param nfa_start_state: El estado inicial del NFA (autómata finito no determinista)
    :param nfa_final_states: Una lista de estados finales en el NFA (autómata finito no determinista).
    Estos son los estados en los que, si el NFA termina en uno de ellos después de procesar la cadena de
    entrada, el NFA acepta la cadena de entrada
    """
    graph = nx.DiGraph()

    for state in nfa_states:
        if state in nfa_final_states:
            graph.add_node(state, final=True)
        else:
            graph.add_node(state)

    for source_state, transitions in nfa_transitions.items():
        for symbol, target_states in transitions.items():
            for target_state in target_states:
                graph.add_edge(source_state, target_state, label=symbol)

    node_positions = nx.spring_layout(graph)
    node_colors = [
        'lightblue' if node in nfa_final_states else 'white' for node in graph.nodes]

    nx.draw_networkx(
        graph,
        node_positions,
        node_color=node_colors,
        node_size=500,
        font_size=12,
        edgecolors='black',
        linewidths=1,
        alpha=0.8
    )

    node_labels = {state: state for state in graph.nodes}
    nx.draw_networkx_labels(graph, node_positions, labels=node_labels)

    edge_labels = {(u, v): data['label']
                   for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        graph, node_positions, edge_labels=edge_labels)

    # Obtén la ventana raíz de tkinter
    root = plt.gcf().canvas.manager.window
    root.title('Automata No Determinista')

    plt.axis('off')
    plt.show()
