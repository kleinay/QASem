
base = document.location.href

function empty_state() {
    return {
        "text": "",
        "tokens": [],
        "lemmas": [],
        "predicate_indices": [],
        "selected_idx": -1,
        "selected_roleset_idx": -1,
        "rolesets": [],
        "questions": [],
        "prototype_question": "",
        "contextualized_question": ""
    }
}

function on_new_text() {
    let text_input = $id("the_text")
    let text = text_input.value || "";
    let text_len = text.length;
    if (text_len > MAX_CHAR_LENGTH) {
        text = text.slice(0, MAX_CHAR_LENGTH);
        text_input.value = text;
        text_len = MAX_CHAR_LENGTH;
    }
    let chars_left = MAX_CHAR_LENGTH - text_len;
    $id("chars_left").innerText = `${chars_left}\t`
}

function $id(id_) {
    return document.getElementById(id_)
}


function show(id) {
    if ($id(id).classList.contains("show")) {
        return;
    }
    let link_id = id.split("_")
    link_id[link_id.length - 1] = "link"
    link_id = link_id.join("_");
    $id(link_id).click()
}

async function hide(id) {
    return new Promise((resolve, reject) => {
        let is_shown = $id(id).classList.contains("show")
        if (!is_shown) {
            console.log("not shown, returning " + id)
            return resolve()
        }
        let link_id = id.split("_")
        link_id[link_id.length - 1] = "link"
        link_id = link_id.join("_");
        setTimeout(() => {
            $id(link_id).click()
            return resolve()
        }, 0)
    })
}


function prepare_post_options(body) {
    return {
        "method": "POST",
        "headers": {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        "body": JSON.stringify(body)
    }
}

function create_token(token) {
    let token_elt = document.createElement("span")
    token_elt.innerText = `${token}`
    return token_elt
}

async function on_apply() {
    the_state = empty_state();
    const text = $id("the_text").value
    if (the_state.text === text) {
        return;
    }

    the_state.text = text;
    show_spinner("apply_loaded");
    the_state.tokens = []
    the_state.rolesets = []
    the_state.questions = []
    the_state.selected_roleset_idx = -1;
    the_state.selected_idx = -1

    const request_body = {"text": the_state.text};
    const resp = await fetch(base + "/api/text", prepare_post_options(request_body));
    const content = await resp.json();

    the_state.tokens = content.tokens;
    the_state.lemmas = content.lemmas;
    the_state.predicate_indices = content.predicate_indices;
    the_state.selected_idx = -1
    the_state.selected_roleset_idx = -1
    await hide_spinner("apply_loaded", 800);
    await hide("input_text_card", 100);
    await hide("sense_disambig_card", 100);


    setTimeout(() => {
        fill_tokens_and_predicates(the_state.tokens, the_state.predicate_indices);
        setTimeout(() => show("main_text_card"), 100)
    }, 100);

}

function show_spinner(spinner_id) {
    $id(spinner_id).classList.add("visible")
    $id(spinner_id).classList.remove("invisible")
}

async function hide_spinner(spinner_id, timeout = 800) {
    let spinner = $id(spinner_id)
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            spinner.classList.add("invisible")
            spinner.classList.remove("visible")
            resolve();
        }, timeout);
    });
}

async function on_prototype_submit() {
    let body = {
        "prototype": the_state.prototype_question,
        "predicate_idx": the_state.selected_idx,
        "lemma": the_state.rolesets[the_state.selected_roleset_idx].lemma,
        "tokens": the_state.tokens,
    }
    let req = prepare_post_options(body)
    show_spinner("prototype_spinner")
    let resp = await fetch(base +"/api/contextualize", req)
    let content = await resp.json()
    await hide_spinner("prototype_spinner", 500)
    the_state.contextualized_question = content.contextualized_question;
    $id("contextualized_question").value = content.contextualized_question;
}

async function on_predicate_selected(evt) {
    const selected_id = evt.target.id;
    let selected_idx = parseInt(selected_id.split("_")[1])
    // if (selected_idx === the_state.selected_idx) {
    //     return
    // }

    the_state.selected_roleset_idx = -1
    the_state.rolesets = []
    the_state.questions = []

    show_spinner("main_text_spinner")
    const req_body = {
        "tokens": the_state.tokens,
        "predicate_idx": selected_idx
    }

    const resp = await fetch(base + "/api/rolesets", prepare_post_options(req_body))
    const content = await resp.json()
    the_state.selected_idx = selected_idx;
    the_state.rolesets = content;
    clear_questions();
    fill_rolesets(the_state.rolesets)
    $id("btn_verb").value = the_state.lemmas[the_state.selected_idx];
    $id("btn_verb_label").innerText = the_state.lemmas[the_state.selected_idx];
    show("sense_disambig_card")
    let is_single_roleset = the_state.rolesets.length === 1;
    if (is_single_roleset) {
        // simulate clicking on the invisible roleset.
        let selected_roleset_idx = 0;
        setTimeout(() => {
            $id(`roleset_${selected_roleset_idx}`).click()
        }, 0);
    } else {
        hide_spinner("main_text_spinner", 300)
    }
}

async function on_roleset_selected(evt) {
    const roleset_id = evt.target.id;
    const roleset_idx = parseInt(roleset_id.split("_")[1]);
    if (roleset_idx === the_state.selected_roleset_idx) {
        return;
    }
    the_state.selected_roleset_idx = roleset_idx;
    for (let idx = 0; idx < the_state.rolesets.length; idx++) {
        $id(`roleset_${idx}`).classList.remove("active")
    }
    evt.target.classList.add("active");
    const roleset = the_state.rolesets[the_state.selected_roleset_idx];
    $id("btn_verb").value = roleset['lemma']
    $id("btn_verb_label").innerText = roleset['lemma']

    const request_body = {
        "lemma": roleset['lemma'],
        "sense_id": roleset['sense_id'],
        "pos": roleset["pos"],
        "tokens": the_state['tokens'],
        "predicate_idx": the_state.selected_idx
    }

    show_spinner("main_text_spinner")
    const resp = await fetch(base + "/api/questions", prepare_post_options(request_body))
    const content = await resp.json()
    the_state.questions = content
    await hide_spinner("main_text_spinner", 800);
    fill_questions(the_state.questions);
}

function on_question_slot_selected() {
    // regenerate the question from all slots
    let filled_slots = {}
    let slot_ids = ['btn_wh', 'btn_aux', 'btn_sbj', 'btn_verb', 'btn_obj', 'btn_prep', 'btn_obj2']

    slot_ids.forEach(slot_id => {
        let slot_buttons = Array.from(document.getElementsByName(slot_id))
        let checked_button = slot_buttons.find(btn => btn.checked)
        if (!checked_button) {
            filled_slots[slot_id] = "";
            return;
        }
        filled_slots[slot_id] = checked_button.value;
    })
    let can_submit = Object.values(filled_slots).every(slot => slot !== "");
    can_submit = can_submit && the_state.selected_roleset_idx !== -1
    $id("btn_submit_prototype").disabled = !can_submit;
    let prototype_question = Object.values(filled_slots).filter(slot => slot !== "" && slot !== "---").join(" ") + "?";
    $id("prototype_question").value = prototype_question
    if (!can_submit) {
        return;
    }
    the_state.prototype_question = prototype_question;
}


function fill_tokens_and_predicates(tokens, predicate_indices) {
    clear_main_text();
    clear_questions();
    clear_rolesets();

    const par = $id("main_text");
    let spans = tokens.map(tok => create_token(tok))
    spans.forEach(span => {
        par.appendChild(span);
        par.appendChild(document.createTextNode(" "))
    })
    for (let pred_idx of predicate_indices) {
        let predicate_span = spans[pred_idx];
        predicate_span.classList.add("predicate_token")
        predicate_span.id = `pred_${pred_idx}`;
    }
    let predicates = document.getElementsByClassName("predicate_token");
    for (let predicate of predicates) {
        predicate.addEventListener("click", on_predicate_selected)
    }
}

function fill_rolesets(rolesets) {
    clear_rolesets();
    const lst_rolesets = $id("lst_rolesets")
    const roleset_template = $id("template_roleset").content
    rolesets.forEach((roleset, i) => {
        let roleset_elt = roleset_template.cloneNode(true);
        lst_rolesets.appendChild(roleset_elt)
        lst_rolesets.lastElementChild.id = `roleset_${i}`
        lst_rolesets.lastElementChild.innerText = `${roleset['lemma']}.${roleset['sense_id']}[${roleset['pos']}]: ` + roleset['role_set_desc'];
        lst_rolesets.lastElementChild.addEventListener("click", on_roleset_selected);
    });

}

function fill_questions(role_questions) {
    clear_questions();

    const tbody_questions = $id("tbody_questions")
    role_questions.forEach(role_question => {
        let role = role_question['role_type'];
        let role_desc = role_question['role_desc'];
        let proto = role_question.questions[0].prototype;
        let generated = role_question.questions[0]['contextualized'];

        let tr = document.createElement("tr");
        let role_elt = document.createElement("th");
        role_elt.innerText = role
        tr.appendChild(role_elt);
        let role_desc_elt = document.createElement("td");
        role_desc_elt.innerText = role_desc;
        tr.appendChild(role_desc_elt);
        let question_elt = document.createElement("td");
        question_elt.innerText = generated;
        tr.appendChild(question_elt);
        let proto_elt = document.createElement("td");
        proto_elt.innerText = proto;
        tr.appendChild(proto_elt);
        tbody_questions.appendChild(tr);
    });
}

function clear_questions() {
    const tbody = $id("tbody_questions")
    while (tbody.firstChild) {
        tbody.removeChild(tbody.firstChild)
    }
}

function clear_rolesets() {
    const par = $id("lst_rolesets");
    while (par.firstChild) {
        par.removeChild(par.firstChild);
    }
}

function clear_main_text() {
    const par = $id("main_text");
    while (par.firstChild) {
        par.removeChild(par.firstChild);
    }
}

let the_state = {}
const MAX_CHAR_LENGTH = 250;
document.addEventListener("DOMContentLoaded", function (event) {
    the_state = empty_state();
    let question_slots = document.getElementsByClassName("question_slot");
    for (let question_slot of question_slots) {
        question_slot.addEventListener("click", on_question_slot_selected);
    }
    on_new_text();
});


