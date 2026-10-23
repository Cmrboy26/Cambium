import json
import os
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from simulator.training.dashboard import RunStore, make_server
from simulator.training.metrics import CATALOG
from simulator.training.replay_requests import (
    ReplayRequestConflict,
    ReplayRequestUnsupported,
    finish_request,
    requested_replay_round,
)


class DashboardChecks(unittest.TestCase):
    def test_page_sections_prioritize_run_overview(self):
        class PageStructure(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
                self.links = []
                self.disclosures = []
                self.active_disclosures = []
                self.opponent_disclosure = None
                self.smoothing_control = None
                self.window_modes = []
                self.window_controls = {}
                self.pages = {}
                self.navigation = {}

            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if "id" in attributes:
                    self.ids.append(attributes["id"])
                if "data-page" in attributes:
                    self.pages[attributes["id"]] = attributes
                if "data-view" in attributes:
                    self.navigation[attributes["data-view"]] = attributes
                if attributes.get("id") == "smooth":
                    self.smoothing_control = (tag, attributes)
                if attributes.get("name") == "window-mode":
                    self.window_modes.append(attributes["value"])
                if attributes.get("id", "").startswith("window-"):
                    self.window_controls[attributes["id"]] = (tag, attributes)
                if tag == "a":
                    self.links.append(attributes.get("href", ""))
                if tag == "details":
                    self.disclosures.append(attributes)
                    self.active_disclosures.append(attributes)
                if attributes.get("id") == "training-opponents" and self.active_disclosures:
                    self.opponent_disclosure = self.active_disclosures[-1]

            def handle_endtag(self, tag):
                if tag == "details":
                    self.active_disclosures.pop()

        page = PageStructure()
        page.feed(Path(__file__).with_name("dashboard").joinpath("index.html").read_text(encoding="utf-8"))
        self.assertEqual(len(page.ids), len(set(page.ids)), "Dashboard IDs must remain unique")
        self.assertIn("replay-request", page.ids)
        self.assertIn("replay-request-status", page.ids)
        ordered = ["run-title", "milestone-browser", "milestone-strip", "milestone-details", "milestone-metrics",
                 "run-identity", "run-context", "stat-update", "stat-remaining",
                   "results", "outcomes-title", "optimization-title", "runtime-title", "curriculum",
                   "world", "run-settings", "recent-updates"]
        positions = [page.ids.index(identity) for identity in ordered]
        self.assertEqual(positions, sorted(positions), "Run identity and metrics must precede detailed sections")
        for target in ("results", "curriculum", "world", "run-settings", "recent-updates"):
            self.assertIn(f"#{target}", page.links)
        self.assertEqual(list(page.navigation), ["overview", "results", "replays", "configuration"])
        self.assertEqual(page.navigation["overview"]["aria-current"], "page")
        for identity in ("results", "world", "run-settings", "recent-updates"):
            self.assertIn("hidden", page.pages[identity], "Only overview is visible on initial load")
        self.assertEqual(page.pages["curriculum"]["data-page"], "overview")
        self.assertEqual(page.pages["recent-updates"]["data-page"], "results")
        self.assertTrue(page.disclosures)
        self.assertTrue(all("open" not in attrs for attrs in page.disclosures), "Detailed configuration starts collapsed")
        self.assertEqual(page.opponent_disclosure, {"id": "training-opponent-details", "class": "opponent-details"},
                 "Opponent statistics must be inside a collapsed, user-toggleable disclosure")
        self.assertEqual(page.smoothing_control, ("input", {"id": "smooth", "type": "range", "min": "0",
                 "max": "1000", "step": "1", "value": "0", "aria-valuetext": "Off"}))
        self.assertIn("smooth-value", page.ids)
        self.assertEqual(page.window_modes, ["latest", "range", "stage", "all"])
        self.assertEqual(page.window_controls["window-size"][1]["type"], "range")
        for bound in ("start", "end"):
            self.assertEqual(page.window_controls[f"window-{bound}"][1]["type"], "number")
            self.assertEqual(page.window_controls[f"window-{bound}-slider"][1]["type"], "range")
        self.assertEqual(page.window_controls["window-stage"][0], "select")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for curriculum DOM checks")
    def test_experiment_stage_evaluation_and_replay_dom(self):
        directory = Path(__file__).with_name("dashboard")
        result = subprocess.run(["node", "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
class Node {
    constructor() {this.children=[]; this.dataset={}; this.attributes={}; this.listeners=new Map(); this.style={setProperty(name,value){this[name]=value;}}; this.hidden=false; this.value='run';}
    set textContent(value) {this.text=String(value); this.children=[];}
    get textContent() {return this.text ?? this.children.map(child=>child.textContent).join(' ');}
    set innerHTML(value) {throw new Error('Untrusted content must never use innerHTML');}
    append(...children) {this.text=undefined; this.children.push(...children);}
    replaceChildren(...children) {this.text=undefined; this.children=children;}
    addEventListener(name,handler) {this.listeners.set(name,handler);}
    removeEventListener(name,handler) {assert.equal(this.listeners.get(name),handler);this.listeners.delete(name);}
    setAttribute(name,value) {this.attributes[name]=value;}
}
const nodes = new Map();
const document = {getElementById(id) {if (!nodes.has(id)) nodes.set(id,new Node()); return nodes.get(id);},
    createElement:()=>new Node(), createDocumentFragment:()=>new Node(), querySelectorAll:()=>[],
    addEventListener() {}, dispatchEvent() {}};
const context = vm.createContext({document, Intl, CustomEvent:class {}, setInterval() {}, cancelAnimationFrame() {}});
vm.runInContext(fs.readFileSync(path.join(process.argv[1],'dashboard.js'),'utf8'), context);
for (const [name,label] of [['missed_attack','Missed attack'],['shield_disabled','Shield disabled'],['win_speed','Win speed'],
    ['arrow_progress','Arrow progress']]) {
    assert.equal(vm.runInContext(`rewardTypes.find(([key])=>key==='${name}')?.[1]`, context),label);
}
const conditionCharts=[];
context.Chart=class {
    constructor(canvas,config) {this.canvas=canvas; this.ctx={save(){}}; this.data=config.data; this.options=config.options; this.plugins=config.plugins; this.destroyed=false; conditionCharts.push(this);for(const plugin of this.plugins)plugin.afterInit?.(this);}
    update() {this.ctx.save();}
    destroy() {
        for(const plugin of this.plugins)plugin.beforeDestroy?.(this);
        this.canvas=null;this.ctx=null;this.destroyed=true;
        for(const plugin of this.plugins)plugin.afterDestroy?.(this);
    }
};
const text = id=>document.getElementById(id).textContent;
const hidden = id=>document.getElementById(id).hidden;
const scene = name=>({name,max_ticks:50,players:[],bounds:{min:[0,0,0],size:[8,4,8]}});
const experiment = {format:'minecraft_sim_experiment',version:1,name:'<img src=x onerror=alert(1)>',seed:17,
    learner:{lr:0.01,epochs:3,minibatch_size:16},policy:{},evaluations:{'<held>':{},unseen:{}},
    stages:[
        {name:'aim',environment_name:'Aim environment',arena_names:['Flat'],scenario:{arenas:[{environment:scene('flat'),weight:1}]},
            controls:{attack:false},opponents:[{id:'target',weight:1}],advance:{updates:2},
            evaluations:{aim_accuracy:{}},
            rewards:{mode:'replace',weights:{good_aim:0.025,death:-10,won_duel:0}}},
        {name:'fight',environment_name:'Fight environment',arena_names:['Hill','Pit'],
            scenario:{arenas:[{environment:scene('hill'),weight:1},{environment:scene('pit'),weight:3}]},
            controls:{},opponents:[{id:'sword',weight:1},{id:'axe',weight:3}],
            advance:{all:[{episodes:10},{any:[{updates:4},{transitions:128}]}]}}
    ]};
const config = {options:{device:'cpu',updates:1000,seed:0},metadata:{environment:scene('deployment'),
    learner:'opponent_pool',reward_weights:{doing_damage:1}},experiment};
context.fixture=config;
vm.runInContext('records=[]; renderConfig(fixture); renderCurriculum();',context);
assert.match(text('curriculum-status'),/Initial stage: aim/);
assert.equal(text('run-title'),experiment.name,'Run identity must be literal text');
assert.match(text('run-identity'),/Seed 17.*Compute cpu.*Learning rate 0.01.*PPO epochs 3.*Minibatch 16/);
assert.match(text('run-context'),/Aim environment.*target.*100%.*attack=false/);
assert.equal(text('run-stage-status'),text('curriculum-status'));
assert.equal(hidden('milestone-browser'),false);
assert.equal(text('milestone-name'),'aim');
assert.equal(hidden('milestone-metrics'),true,'Update progress must not get a redundant graph');
assert.equal(conditionCharts.length,0);
assert.equal(text('milestone-position'),'Stage 1 of 2');
assert.match(text('milestone-current'),/Initial stage: aim/);
assert.equal(document.getElementById('milestone-previous').disabled,true);
const milestoneSlots = () => document.getElementById('milestone-strip').children[0].children;
assert.equal(milestoneSlots().length,3);
assert.equal(milestoneSlots()[1].attributes['aria-pressed'],'true','The selected stage must occupy the center slot');
assert.equal(milestoneSlots()[1].attributes['aria-current'],'step');
assert.match(text('milestone-reward-source'),/Saved stage/);
assert.match(text('milestone-reward-count'),/2 active.*3 recorded/);
assert.match(text('milestone-rewards'),/Death.*-10.*Good aim.*\+0.025/);
const rewardRows = document.getElementById('milestone-rewards').children[0].children;
assert.equal(rewardRows[0].dataset.sign,'negative');
assert.equal(rewardRows[1].dataset.sign,'positive');
assert.equal(rewardRows[0].children[2].children[0].style.width,'100%');
assert.equal(rewardRows[1].children[2].children[0].style.width,'0.25%');
assert.equal(document.getElementById('milestone-controls').dataset.mode,'fixed');
assert.match(text('milestone-rewards-inactive'),/Win.*0/);
assert.doesNotMatch(text('milestone-rewards'),/Damage dealt/,'Never use deployment rewards for a stage');
document.getElementById('milestone-next').onclick();
assert.equal(conditionCharts.length,2,'Nested conditions omit updates but retain episode and transition charts');
assert.ok(conditionCharts.every(chart=>chart.plugins.some(plugin=>plugin.id==='legendSolo')),
    'Milestone value/threshold legends must support soloing too');
assert.doesNotMatch(text('milestone-metric-charts'),/Completed updates/);
assert.equal(text('milestone-name'),'fight');
assert.equal(text('milestone-position'),'Stage 2 of 2');
assert.equal(milestoneSlots()[1].attributes['aria-pressed'],'true');
assert.equal(milestoneSlots()[0].attributes['aria-current'],'step','Current and selected stage are separate states');
assert.match(text('milestone-environment'),/Fight environment.*Hill.*Pit/);
assert.match(text('milestone-opponents'),/sword.*25%.*axe.*75%/);
const opponentRows = document.getElementById('milestone-opponents').children[0].children;
assert.equal(opponentRows[0].children[2].children[0].style.width,'25%');
assert.equal(opponentRows[1].children[2].children[0].style.width,'75%');
assert.match(text('milestone-reward-empty'),/Not recorded/);
assert.equal(text('milestone-rewards'),'','Changing stages must clear reward rows');
assert.equal(text('milestone-rewards-inactive'),'');
assert.match(text('milestone-controls'),/All learned/);
assert.equal(document.getElementById('milestone-controls').dataset.mode,'learned');
assert.match(text('milestone-observation'),/No rollout recorded/);
assert.equal(document.getElementById('milestone-next').disabled,true);
assert.match(text('milestone-current'),/Initial stage: aim/,'Browsing must not change the current stage');
vm.runInContext('renderCurriculum();',context);
assert.equal(text('milestone-name'),'fight','Live refresh must preserve the browsed stage');
document.getElementById('milestone-return').onclick();
assert.equal(text('milestone-name'),'aim');
assert.equal(conditionCharts.every(chart=>chart.destroyed),true,'Returning to an update-only stage must release charts');
assert.equal(vm.runInContext('milestoneCharts.size',context),0,'Chart destruction must finish clearing the stage cache');
document.getElementById('milestone-next').onclick();
vm.runInContext('renderCurriculum();',context);
assert.ok(conditionCharts.slice(-2).every(chart=>chart.ctx),'Refreshing a revisited stage must not reuse destroyed contexts');
document.getElementById('milestone-previous').onclick();
assert.match(text('stage-context'),/Aim environment.*Flat.*target.*100%.*attack=false/);
assert.match(text('config'),/Seed 17.*Learning rate 0.01/);
assert.equal(hidden('experiment-details'),false);
assert.equal(document.getElementById('experiment-details').open,undefined,'Research details must default collapsed');
assert.equal(text('curriculum-title'),experiment.name,'Labels must be literal text');
assert.deepEqual(JSON.parse(text('experiment-json')),experiment);
assert.match(text('environment-name'),/Deployment snapshot/);
assert.match(text('evaluations'),/<held>.*--.*unseen/);
assert.match(text('evaluations'),/aim_accuracy/);
vm.runInContext(`records=[{update:2,stage:'aim',stage_index:0,next_stage_index:1,
    stage_progress:{updates:2,episodes:3,transitions:64},completed_curriculum:false,
    curriculum_boundary:{reset_worlds:2,next_stage:'fight'},
    evaluations:{'<held>':{win_rate:0.75,reward_mean:-1.5,episodes:4}}}]; renderCurriculum();`,context);
assert.match(text('curriculum-status'),/Upcoming stage: fight.*latest rollout is from aim/);
assert.equal(text('milestone-name'),'fight','Follow the stage selected at the recorded boundary');
assert.match(text('milestone-current'),/Current stage: fight/);
assert.match(text('milestone-observation'),/No rollout recorded/,'Do not attribute the previous stage progress to the next stage');
document.getElementById('milestone-previous').onclick();
assert.equal(text('milestone-name'),'aim');
assert.match(text('milestone-state'),/Completed/);
assert.match(text('milestone-progress'),/2.*updates.*3.*episodes.*64.*transitions/);
assert.equal(document.getElementById('milestone-progress-bar').hidden,false);
assert.equal(document.getElementById('milestone-progress-bar').max,2);
assert.equal(document.getElementById('milestone-progress-bar').value,2);
assert.equal(hidden('milestone-metrics'),true);
assert.equal(text('milestone-metric-charts'),'','Update-only stages must clear the metric section');
document.getElementById('milestone-return').onclick();
assert.match(text('run-context'),/Aim environment.*target.*100%/,'Boundary summary must describe the latest rollout, not the upcoming stage');
assert.match(text('run-budget'),/updates: 2 \/ 2/);
assert.match(text('stage-summary'),/Latest rollout stage aim.*updates: 2 \/ 2.*Not complete.*Upcoming stage fight/);
assert.match(text('stage-context-title'),/Latest rollout conditions — aim/);
assert.equal(hidden('upcoming-stage'),false);
assert.match(text('upcoming-context'),/Fight environment.*Hill.*Pit.*sword \(25%\).*axe \(75%\).*all heads learned/);
assert.match(text('stage-timeline'),/2 aim.*2 worlds reset → fight/);
assert.match(text('evaluations'),/<held> 2 aim 75% -1.5 4/);
vm.runInContext(`records.push({update:3,stage:'fight',stage_index:1,next_stage_index:1,stage_progress:{updates:1,episodes:4,transitions:32},
    completed_curriculum:false,evaluations:{}}); renderCurriculum();`,context);
assert.match(text('curriculum-status'),/Active stage.*fight/);
assert.match(text('run-context'),/Fight environment.*sword \(25%\).*axe \(75%\)/);
assert.equal(hidden('upcoming-stage'),true);
assert.match(text('stage-summary'),/ALL \(episodes: 4 \/ 10; ANY \(updates: 1 \/ 4; transitions: 32 \/ 128\)\)/);
assert.match(text('evaluations'),/<held> 2 aim 75% -1.5 4/,'Periodic results must retain their original update/stage');
vm.runInContext(`records.at(-1).evaluations={'<held>':{win_rate:0.75,reward_mean:-1.5,episodes:4,
    worlds:4,seconds:0.125,snapshot_update:2,stage:'aim',stage_index:0}}; renderCurriculum();`,context);
assert.match(text('evaluations'),/<held> 2 aim 75% -1.5 4 4 0.125/,'Repeated results must retain snapshot metadata');
vm.runInContext(`records.push({update:4,stage:'fight',stage_index:1,next_stage_index:1,
    stage_progress:{updates:2,episodes:10,transitions:128},completed_curriculum:true}); renderCurriculum();`,context);
assert.equal(text('curriculum-status'),'Curriculum complete','Completion is explicit, not inferred from final stage');
assert.equal(text('run-stage-status'),'Curriculum complete');
assert.match(text('milestone-current'),/Curriculum complete/);
assert.match(text('milestone-state'),/Completed/);
assert.equal(hidden('upcoming-stage'),true,'Complete wins over stale next stage');
assert.match(text('stage-timeline'),/3–4 fight.*Curriculum complete/);
vm.runInContext(`records=[{update:1}]; renderConfig({options:{seed:5},metadata:{environment:{name:'Historical'}}}); renderCurriculum();`,context);
assert.equal(hidden('curriculum'),true);
assert.equal(hidden('milestone-browser'),true);
assert.equal(hidden('milestone-metrics'),true);
assert.equal(conditionCharts.at(-1).destroyed,true,'Changing to a legacy run must release charts');
assert.equal(hidden('experiment-details'),true);
assert.equal(text('environment-name'),'Historical');
assert.equal(text('run-title'),'Historical');
assert.match(text('run-identity'),/Seed 5/);
assert.match(text('run-context'),/Historical/);
assert.doesNotMatch(text('run-context'),/sword|axe|Fight environment/,'Switching to historical runs must clear stage context');
vm.runInContext(`records=[]; renderConfig({plan:{name:'Old plan',scenario:{base:'missing.json'},controls:{fixed:{attack:true}},
    stages:[{name:'old'}]}}); renderCurriculum();`,context);
assert.match(text('curriculum-status'),/Initial stage: old/);
assert.equal(text('milestone-name'),'old','Changing runs must reset the browsed stage');
assert.equal(document.getElementById('milestone-next').disabled,true);
assert.equal(document.getElementById('milestone-previous').disabled,true);
assert.match(text('milestone-reward-empty'),/Not recorded/,'Unresolved authoring must not borrow inherited rewards');
vm.runInContext(`renderMilestoneRewards({task_override:{rewards:{good_aim:0.00000001,death:-2}}});`,context);
assert.match(text('milestone-reward-source'),/Task override/);
assert.match(text('milestone-rewards'),/1e-8/,'Small nonzero weights must not round to zero');
vm.runInContext(`renderMilestoneRewards({rewards:{mode:'replace',weights:{}}});`,context);
assert.match(text('milestone-reward-empty'),/No reward terms configured/);
vm.runInContext(`renderMilestoneRewards({rewards:{mode:'replace',weights:{death:0}}});`,context);
assert.match(text('milestone-reward-empty'),/All recorded reward weights are zero/);
assert.match(text('milestone-rewards-inactive'),/Death.*0/);
vm.runInContext(`renderMilestoneRewards({scenario:{arenas:[
    {environment:{name:'Hill',reward_weights:{death:-2}}},
    {environment:{name:'Pit',reward_weights:{doing_damage:3}}}]}});`,context);
assert.match(text('milestone-rewards'),/Hill.*Death.*-2.*Pit.*Damage dealt.*\+3/,'Historical arena rewards must retain their arena identity');
assert.match(text('stage-context'),/Fixed controls Not recorded/,'Never resolve authoring inheritance');
vm.runInContext(`const longer = JSON.parse(JSON.stringify(fixture));
    longer.experiment.name='Four stage curriculum';
    longer.experiment.stages.push({...longer.experiment.stages[1],name:'pursuit'},
        {...longer.experiment.stages[1],name:'combat'});
    records=[{update:3,stage:'fight',stage_index:1,stage_progress:{updates:1,episodes:2,transitions:64}}];
    renderConfig(longer); renderCurriculum();`,context);
assert.equal(text('milestone-name'),'fight');
assert.match(milestoneSlots()[0].textContent,/aim/);
assert.match(milestoneSlots()[2].textContent,/pursuit/);
assert.equal(document.getElementById('milestone-progress-bar').hidden,true,'Combined criteria must not become a fake percentage');
const stageKey = key => document.getElementById('milestone-strip').onkeydown({key,preventDefault(){}});
stageKey('End');
assert.equal(text('milestone-name'),'combat');
stageKey('ArrowLeft');
assert.equal(text('milestone-name'),'pursuit');
vm.runInContext(`records.push({update:4,stage:'pursuit',stage_index:2,stage_progress:{updates:1,episodes:1,transitions:32}});
    renderCurriculum();`,context);
assert.equal(text('milestone-name'),'pursuit','A browsed stage remains selected as training advances');
assert.match(text('milestone-current'),/Current stage: pursuit/);
assert.equal(document.getElementById('milestone-return').hidden,true);
stageKey('Home');
assert.equal(text('milestone-name'),'aim');
assert.match(text('milestone-current'),/Current stage: pursuit/);
document.getElementById('milestone-return').onclick();
vm.runInContext(`records=[{update:5,stage:'unknown'}]; renderCurriculum();`,context);
assert.equal(text('milestone-current'),'Current stage not recorded');
assert.equal(text('milestone-state'),'Status not recorded');
assert.equal(document.getElementById('milestone-progress-bar').hidden,true);
vm.runInContext(`const monitored=JSON.parse(JSON.stringify(fixture));
    monitored.experiment.stages[0].advance={evaluation:{suite:'aim_accuracy',metric:'reward_mean',at_least:6}};
    records=[{update:25,stage_index:0,stage_progress:{updates:25},evaluations:{aim_accuracy:{
        reward_mean:5.5,snapshot_update:25,stage_index:0}}},
        {update:26,stage_index:0,stage_progress:{updates:26},evaluations:{aim_accuracy:{
        reward_mean:5.5,snapshot_update:25,stage_index:0}}}];
    renderConfig(monitored); renderCurriculum();`,context);
const accuracyChart=conditionCharts.at(-1);
assert.equal(accuracyChart.data.datasets[0].data.length,1,'Carried evaluation results are not new points');
assert.equal(accuracyChart.data.datasets[0].data[0].x,25);
assert.equal(accuracyChart.data.datasets[0].pointRadius,3,'Single evaluations must remain visible');
assert.deepEqual(Array.from(accuracyChart.data.datasets[1].borderDash),[6,4]);
assert.equal(accuracyChart.data.datasets[1].data[0].y,6);
assert.match(text('milestone-metric-charts'),/Mean episode return.*5.5 \/ 6.*aim_accuracy/);
const chartCount=conditionCharts.length;
vm.runInContext('renderCurriculum();',context);
assert.equal(conditionCharts.length,chartCount,'Live refresh must reuse charts');
vm.runInContext(`records.push({update:50,stage_index:0,stage_progress:{updates:50},
    evaluations:{aim_accuracy:{reward_mean:6.2,win_rate:0.8,snapshot_update:50,stage_index:0}}});
    renderCurriculum();`,context);
assert.equal(conditionCharts.length,chartCount,'New measurements must update the existing chart');
assert.equal(accuracyChart.data.datasets[0].data.length,2);
assert.equal(accuracyChart.options.scales.x.max,50,'History must grow with stage progress');
assert.match(text('milestone-metric-charts'),/6.2 \/ 6/);
vm.runInContext(`monitored.experiment.stages[0].advance.evaluation.metric='win_rate';
    monitored.experiment.stages[0].advance.evaluation.at_least=0.75; renderCurriculum();`,context);
assert.match(text('milestone-metric-charts'),/Win rate.*80% \/ 75%/);
assert.equal(conditionCharts.at(-1).options.scales.y.suggestedMax,1);
vm.runInContext('records=[]; renderCurriculum();',context);
assert.match(text('milestone-metric-charts'),/Awaiting evaluation/);
assert.equal(conditionCharts.at(-1).data.datasets[0].data.length,0);
assert.equal(conditionCharts.at(-1).data.datasets[1].data[0].y,0.75,'The threshold remains visible before any evaluation');
document.getElementById('milestone-next').onclick();
assert.equal(accuracyChart.destroyed,true);
// Replay metadata is independent of self-play, and equipment remains concrete.
const arenaSource=fs.readFileSync(path.join(process.argv[1],'arena.js'),'utf8');
vm.runInContext(arenaSource.slice(arenaSource.indexOf('function renderReplayTraining'),arenaSource.indexOf('\nclass ArenaViewer')),context);
vm.runInContext(`renderReplayTraining({stage:'<aim>',opponent:{id:'frozen',type:'checkpoint'},
    controls:{fixed:{movement:'neutral',attack:false}},policy_update:2,completed_rounds:4});`,context);
assert.equal(hidden('replay-training'),false);
assert.match(text('replay-training'),/Stage <aim>.*Opponent.*frozen.*movement="neutral", attack=false.*Policy update 2/);
vm.runInContext('renderReplayTraining(null);',context);
assert.equal(hidden('replay-training'),true);
assert.equal(text('replay-training'),'');
""", str(directory)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for world-view checks")
    def test_world_view_toggle(self):
        script = Path(__file__).with_name("dashboard") / "arena.js"
        result = subprocess.run(["node", "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const nodes = new Map();
const node = () => ({dataset:{}, listeners:{}, addEventListener(type, callback) {this.listeners[type]=callback;},
    replaceChildren() {}, append() {}, setAttribute() {}});
const document = {getElementById(id) {if (!nodes.has(id)) nodes.set(id,node()); return nodes.get(id);},
    querySelectorAll:()=>[], createElement:node, addEventListener() {}};
const context = vm.createContext({document, setInterval() {}, cancelAnimationFrame() {}, resizeCount:0});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').replace(/^import .*;\n/gm,''),context);
vm.runInContext('viewer={resize:()=>globalThis.resizeCount++}; playing=true; tick=42;',context);
const checkbox = document.getElementById('show-world');
checkbox.checked = false;
checkbox.listeners.change({target:checkbox});
assert.equal(document.getElementById('world-view').hidden,true);
assert.equal(vm.runInContext('playing',context),false);
assert.equal(vm.runInContext('tick',context),42);
checkbox.checked = true;
checkbox.listeners.change({target:checkbox});
assert.equal(document.getElementById('world-view').hidden,false);
assert.equal(context.resizeCount,1);
assert.equal(vm.runInContext('playing',context),false);
assert.equal(vm.runInContext('tick',context),42);
""", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for weapon transform checks")
    def test_weapon_swing_is_forward_from_main_hand(self):
        directory = Path(__file__).with_name("dashboard")
        result = subprocess.run(["node", "--input-type=module", "-e", r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
const read = (name) => fs.readFileSync(path.join(process.argv[1], name), 'utf8');
const moduleUrl = (source) => 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
const coreUrl = moduleUrl(read('three.core.min.js'));
const THREE = await import(moduleUrl(read('three.module.min.js').replaceAll('./three.core.min.js', coreUrl)));
const { playerLoadout } = await import(moduleUrl(read('arena-data.js')));
const source = read('arena.js');
const classEnd = source.indexOf('\n}\n', source.indexOf('class ArenaViewer')) + 2;
const context = {THREE, playerLoadout, document: {getElementById: () => ({}),
    createElement: () => ({getContext: () => ({fillText() {}})})}};
vm.runInNewContext(source.slice(0, classEnd).replace(/^import .*;\n/gm, '') +
    '\nglobalThis.Viewer = ArenaViewer;', context);
for (const weapon of ['minecraft:iron_sword', 'minecraft:netherite_axe']) {
    for (const yaw of [0, 90, 180, -90]) {
        const viewer = Object.create(context.Viewer.prototype);
        viewer.players = new THREE.Group();
        viewer.draw = () => {};
        const player = {position: [5, 1, -3], yaw, loadout: {weapons: [weapon]}};
        viewer.setPlayers([player]);
        const actor = viewer.players.children[0];
        const weaponGroup = actor.userData.weaponGroup;
        const tip = weaponGroup.children[1];
        const localPoint = (object) => actor.worldToLocal(object.getWorldPosition(new THREE.Vector3()));
        const resting = localPoint(tip);
        const pivot = weaponGroup.getWorldPosition(new THREE.Vector3());
        assert.ok(weaponGroup.position.x < 0 && weaponGroup.position.y > 0.7, 'Pivot must be at the weapon hand');
        viewer.setPlayers([{...player, controls: {attack: true}}]);
        const attacking = localPoint(tip);
        assert.ok(attacking.z > resting.z, `${weapon} at yaw ${yaw} must swing forward`);
        assert.ok(attacking.x < 0, 'Weapon must stay on the main-hand side');
        assert.ok(pivot.distanceTo(weaponGroup.getWorldPosition(new THREE.Vector3())) < 1e-9);
        viewer.setPlayers([player]);
        assert.ok(localPoint(tip).distanceTo(resting) < 1e-9, 'Weapon must return to rest');
    }
}
""", str(directory)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for replay interpolation checks")
    def test_replay_interpolation(self):
        script = Path(__file__).with_name("dashboard") / "arena-data.js"
        result = subprocess.run(["node", "--input-type=module", "-e", r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
const { replaySample, replayPlayers } = await import('data:text/javascript;base64,' +
    Buffer.from(fs.readFileSync(process.argv[1], 'utf8')).toString('base64'));
const actor = (position, yaw, health) => ({position, yaw, health, pitch:0});
const frames = [{tick:0,actors:[actor([0,1,0],179,20),actor([4,1,0],0,20)]},
    {tick:2,actors:[actor([2,1,0],-179,10),actor([2,1,0],0,15)]}];
const middle = replaySample(frames, 1);
assert.equal(middle.index, 0);
assert.deepEqual(middle.actors[0].position, [1,1,0]);
assert.equal(middle.actors[0].yaw, 180);
assert.equal(middle.actors[0].health, 20);
assert.equal(replaySample(frames, 2).actors[0].health, 10);
assert.deepEqual(replaySample(frames, -2).actors[0].position, [0,1,0]);
assert.deepEqual(replaySample(frames, 100).actors[0].position, [2,1,0]);
assert.equal(replaySample([], 0), null);
assert.deepEqual(middle.projectiles, []);
const shot = {id: 0, owner: 1, age: 1, position: [0,2,0], velocity: [3,0,0], in_ground: false};
frames[0].projectiles = [shot];
frames[1].projectiles = [{...shot, age: 3, position: [6,2,0]}];
assert.deepEqual(replaySample(frames, 1).projectiles[0].position, [3,2,0]);
frames[1].projectiles[0].age = 0;
assert.deepEqual(replaySample(frames, 1).projectiles[0].position, [0,2,0], 'Reused slots must not interpolate unrelated shots');
frames[0].projectiles[0].in_ground = true;
frames[1].projectiles[0].age = 3;
assert.deepEqual(replaySample(frames, 1).projectiles[0].position, [0,2,0], 'Embedded arrows remain stationary');
const scenario = {players:[{loadout:{weapons:['minecraft:iron_sword','minecraft:netherite_axe'],
    selected:0,offhand:'minecraft:shield',armor:{helmet:'minecraft:diamond_helmet'}}}]};
const [player] = replayPlayers(scenario, [{...frames[0].actors[0],weapon:'minecraft:netherite_axe',shield_equipped:false}]);
assert.equal(player.loadout.selected, 1);
assert.equal(player.loadout.offhand, 'minecraft:air');
assert.equal(player.loadout.armor.helmet, 'minecraft:diamond_helmet');
assert.equal(replayPlayers(scenario,[frames[0].actors[0]])[0].loadout.selected,0);
""", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for milestone history checks")
    def test_milestone_condition_history(self):
        script = Path(__file__).with_name("dashboard") / "dashboard.js"
        result = subprocess.run(["node", "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({document:{getElementById:()=>({dataset:{}})},Intl});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const evaluate = code=>JSON.parse(JSON.stringify(vm.runInContext(code,context)));
vm.runInContext(`const experiment={evaluations:{combat:{every_updates:2}},stages:[
    {name:'aim',evaluations:{accuracy:{every_updates:2}},advance:{all:[{updates:4},{any:[
        {evaluation:{suite:'accuracy',metric:'reward_mean',at_least:6}},
        {evaluation:{suite:'combat',metric:'win_rate',at_least:0.75}}]}]}},
    {name:'fight',advance:{episodes:100}}]};
const result=(snapshot,value)=>({snapshot_update:snapshot,stage:'aim',stage_index:0,reward_mean:value,episodes:64});
records=[
    {update:10,stage_index:0,stage_progress:{updates:1},evaluations:{}},
    {update:11,stage_index:0,stage_progress:{updates:2},evaluations:{accuracy:result(11,4)}},
    {update:12,stage_index:0,stage_progress:{updates:3},evaluations:{accuracy:result(11,4)}},
    {update:13,stage_index:0,stage_progress:{updates:4},evaluations:{accuracy:result(13,6.2)}},
    {update:14,stage_index:1,stage_progress:{updates:1,episodes:2},evaluations:{accuracy:result(13,6.2)}}];`,context);
assert.equal(evaluate('milestoneMetricHistory(experiment.stages[0],experiment).length'),3);
assert.deepEqual(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[1].data'),
    [{x:2,y:4,update:11},{x:4,y:6.2,update:13}]);
assert.deepEqual(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[1].thresholdData'),
    [{x:0,y:6},{x:4,y:6}]);
assert.equal(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[1].xLabel'),'Stage update');
assert.deepEqual(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[0].data.map(point=>point.y)'),[1,2,3,4]);
assert.deepEqual(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[2].data'),[]);
assert.deepEqual(evaluate('milestoneMetricHistory(experiment.stages[1],experiment)[0].data'),[{x:1,y:2,update:14}]);
vm.runInContext(`records.push({update:15,stage_index:0,stage_progress:{updates:5},
    evaluations:{accuracy:{...result(15,99),stage:'fight',stage_index:1}}});`,context);
assert.equal(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[1].data.length'),2);
vm.runInContext(`records=records.slice(0,4); for(const row of records) {
    if(row.evaluations.accuracy) delete row.evaluations.accuracy.snapshot_update;
}`,context);
assert.deepEqual(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[1].data'),
    [{x:2,y:4,update:11},{x:4,y:6.2,update:13}]);
vm.runInContext(`delete experiment.stages[0].evaluations.accuracy.every_updates;
    for(const row of records) delete row.stage_progress;`,context);
assert.deepEqual(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[1].data'),
    [{x:11,y:4,update:11},{x:13,y:6.2,update:13}]);
assert.equal(evaluate('milestoneMetricHistory(experiment.stages[0],experiment)[1].xLabel'),'PPO update');
assert.equal(evaluate('milestoneMetricHistory(experiment.stages[0],experiment,[])[1].data.length'),0);
assert.deepEqual(evaluate('advancementMetrics({all:[{updates:2},{any:[{episodes:10},{transitions:100}]}]})')
    .map(condition=>condition.metric),['updates','episodes','transitions']);
""", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for dashboard chart checks")
    def test_run_picker_labels_navigation_and_legend_solo(self):
        script = Path(__file__).with_name("dashboard") / "dashboard.js"
        result = subprocess.run(["node", "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Node {
    constructor(tag='div') {this.tag=tag;this.children=[];this.dataset={};this.attributes={};this.listeners={};this.value='';this.scrollTop=0;}
    set textContent(value) {this.text=String(value);this.children=[];}
    get textContent() {return this.text ?? this.children.map(child=>child.textContent).join(' ');}
    set innerHTML(value) {throw new Error('Unsafe HTML');}
    append(...children) {this.text=undefined;this.children.push(...children);}
    replaceChildren(...children) {this.text=undefined;this.children=children.flatMap(child=>child.tag==='fragment'?child.children:[child]);}
    addEventListener(name,fn) {this.listeners[name]=fn;}
    removeEventListener(name,fn) {assert.equal(this.listeners[name],fn);delete this.listeners[name];}
    setAttribute(name,value) {this.attributes[name]=value;}
    removeAttribute(name) {delete this.attributes[name];}
    contains(node) {return node===this || this.children.some(child=>child.contains(node));}
    querySelectorAll(tag) {return this.children.flatMap(child=>[...(child.tag===tag?[child]:[]),...child.querySelectorAll(tag)]);}
    closest() {return this.dataset.page?this:null;}
    focus() {document.activeElement=this;}
}
const nodes=new Map();
const pages=['overview','results','world','run-settings','curriculum','recent-updates'].map((id,index)=>{
    const node=new Node();node.dataset.page=['overview','results','replays','configuration','overview','results'][index];
    nodes.set(id,node);return node;
});
nodes.get('curriculum').dataset.available='false';
const links=['overview','results','replays','configuration'].map(view=>{
    const node=new Node('a');node.dataset.view=view;return node;
});
const document={activeElement:null,events:[],
    getElementById(id){if(!nodes.has(id))nodes.set(id,new Node());return nodes.get(id);},
    createElement:tag=>new Node(tag),createDocumentFragment:()=>new Node('fragment'),
    querySelectorAll(selector){return selector==='[data-page]'?pages:selector==='[data-view]'?links:[];},
    addEventListener(){},dispatchEvent(event){this.events.push(event);}};
const saved=new Map();
const localStorage={getItem:key=>saved.get(key)??null,setItem:(key,value)=>saved.set(key,value)};
const context=vm.createContext({document,Intl,localStorage,window:{location:{hash:''},addEventListener(){}},
    CustomEvent:class {constructor(type,options){this.type=type;this.detail=options.detail;}}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const evaluate=code=>JSON.parse(JSON.stringify(vm.runInContext(code,context)));
const get=id=>document.getElementById(id);
vm.runInContext(`runEntries=[
    {id:'a',name:'job-20',path:'/runs/a',modified:100,experiment_name:'Shield axe',seed:42,hidden_dim:512,lr:.0003},
    {id:'b',name:'job-9',path:'/runs/b',modified:200,experiment_name:'Baseline',seed:7,hidden_dim:256,lr:.001}
]; element('run').value='a'; initRunPicker(); renderRunPicker();`,context);
assert.deepEqual(evaluate('sortRuns(runEntries).map(run=>run.id)'),['b','a'],'Activity date must override directory name');
assert.deepEqual(evaluate("searchRuns(runEntries,'WIDTH 512 seed 42 3e-4').map(run=>run.id)"),['a']);
assert.deepEqual(evaluate("searchRuns(runEntries,'unknown').map(run=>run.id)"),[]);
assert.match(get('run-current').textContent,/Shield axe/);
assert.match(get('run-current-meta').textContent,/Width 512.*LR 0.0003.*Updated/);
get('run-search').value='missing';
vm.runInContext('renderRunPicker();',context);
assert.match(get('run-options').textContent,/No matching runs/);
assert.equal(get('run').value,'a','Filtering must never change the selected run');
get('run-search').value='';
vm.runInContext('renderRunPicker();',context);
get('run-display-name').value='<img src=x onerror=alert(1)>';
get('run-notes').value='Wider network experiment';
vm.runInContext('saveRunLabel({preventDefault(){}});',context);
assert.equal(get('run-title').textContent,'<img src=x onerror=alert(1)>','Labels must remain literal text');
assert.match(get('run-label-status').textContent,/Saved in this browser/);
assert.deepEqual(evaluate("searchRuns(runEntries,'wider network').map(run=>run.id)"),['a']);
assert.equal(JSON.parse([...saved.values()][0])['/runs/a'].notes,'Wider network experiment');
vm.runInContext('runPreferences={};loadRunPreferences();',context);
assert.equal(evaluate("runTitle(runEntries[0])"),'<img src=x onerror=alert(1)>','Labels must survive a reload');
get('run-display-name').value='Unsaved draft';
vm.runInContext('renderRunPicker();',context);
assert.equal(get('run-display-name').value,'Unsaved draft','Polling must not overwrite an in-progress label edit');
context.localStorage.setItem=()=>{throw new Error('quota exceeded');};
vm.runInContext('saveRunLabel({preventDefault(){}});',context);
assert.match(get('run-label-status').textContent,/Could not save.*quota exceeded/);
assert.notEqual(evaluate("runTitle(runEntries[0])"),'Unsaved draft','Failed writes must not look saved');
context.localStorage.setItem=(key,value)=>saved.set(key,value);
get('run-display-name').value='';
get('run-notes').value='';
vm.runInContext('saveRunLabel({preventDefault(){}});',context);
assert.equal(evaluate("runTitle(runEntries[0])"),'Shield axe','Clearing the label restores the recorded name');
saved.set('cambium.run-labels.v1','[]');
vm.runInContext('loadRunPreferences();',context);
assert.match(get('run-label-status').textContent,/Could not load.*invalid format/);
vm.runInContext('showPage("results");',context);
assert.equal(get('results').hidden,false);
assert.equal(get('recent-updates').hidden,false);
assert.equal(get('overview').hidden,true);
assert.equal(get('world').hidden,true);
assert.equal(links[1].attributes['aria-current'],'page');
assert.equal(links[0].attributes['aria-current'],undefined);
assert.equal(document.events.at(-1).detail.page,'results');
vm.runInContext('showPage("overview");',context);
assert.equal(get('curriculum').hidden,true,'Unavailable curriculum must not be exposed by navigation');
context.window.location.hash='#run-settings';
vm.runInContext('navigateHash();',context);
assert.equal(get('run-settings').hidden,false,'Direct links must activate the correct page');
context.window.location.hash='#unknown';
vm.runInContext('navigateHash();',context);
assert.equal(get('overview').hidden,false);
vm.runInContext('let refreshCount=0; refresh=()=>{refreshCount++;};',context);
get('run-options').children[0].focus();
const focused=document.activeElement.dataset.runId;
vm.runInContext('runEntries[0].modified++; renderRunPicker();',context);
assert.equal(document.activeElement.dataset.runId,focused,'Polling must preserve keyboard focus in the picker');
get('run-picker').listeners.keydown({key:'Escape',preventDefault(){},target:get('run-search')});
assert.equal(get('run-picker').open,false);
assert.equal(document.activeElement,get('run-picker-toggle'));
vm.runInContext('selectRun("b");',context);
assert.equal(get('run').value,'b');
assert.equal(get('run-display-name').value,'');
assert.equal(evaluate('refreshCount'),1);
context.Chart={helpers:{getRelativePosition:event=>({x:event.x,y:event.y})}};
const chart={canvas:new Node('canvas'),data:{datasets:[{},{},{}]},visible:[true,true,true],updates:0,
    isDatasetVisible(index){return this.visible[index];},setDatasetVisibility(index,visible){this.visible[index]=visible;},
    update(){this.updates++;},legend:{options:{display:true},
        legendHitBoxes:[{left:0,top:50,width:30,height:10},{left:40,top:50,width:30,height:10},{left:0,top:70,width:30,height:10}],
        legendItems:[{datasetIndex:0},{datasetIndex:1},{datasetIndex:2}]}};
context.chart=chart;
vm.runInContext('legendSolo.afterInit(chart);',context);
let prevented=0;
chart.canvas.listeners.contextmenu({x:45,y:55,preventDefault(){prevented++;}});
assert.deepEqual(chart.visible,[false,true,false]);
chart.canvas.listeners.contextmenu({x:45,y:55,preventDefault(){prevented++;}});
assert.deepEqual(chart.visible,[true,true,true],'Right-clicking the sole visible entry must restore all');
chart.visible=[true,false,true];
chart.canvas.listeners.contextmenu({x:5,y:75,preventDefault(){prevented++;}});
assert.deepEqual(chart.visible,[false,false,true],'Wrapped legend rows must work');
chart.canvas.listeners.contextmenu({x:5,y:55,preventDefault(){prevented++;}});
assert.deepEqual(chart.visible,[true,false,false],'Right-clicking another hidden entry switches the solo');
chart.canvas.listeners.contextmenu({x:300,y:20,preventDefault(){prevented++;}});
assert.equal(prevented,4,'Right-clicks outside the legend must retain the native menu');
chart.visible=[false,false,false];
vm.runInContext('soloLegend(chart,1);',context);
assert.deepEqual(chart.visible,[false,true,false],'Soloing must also work when every series is hidden');
const canvas=chart.canvas;
vm.runInContext('soloLegend(chart,20); legendSolo.beforeDestroy?.(chart); chart.canvas=null; legendSolo.afterDestroy?.(chart);',context);
assert.equal(canvas.listeners.contextmenu,undefined,'Destroyed charts must release their handler before Chart.js clears the canvas');
vm.runInContext('legendSolo.beforeDestroy?.(chart); legendSolo.afterDestroy?.(chart);',context);
chart.canvas=canvas;
chart.data.datasets=[{}];chart.visible=[true];
vm.runInContext('soloLegend(chart,0);legendSolo.afterInit(chart);',context);
assert.deepEqual(chart.visible,[true]);
assert.equal(chart.canvas.listeners.contextmenu,undefined,'Single-series charts retain normal context menus');
""", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for dashboard chart checks")
    def test_chart_math_and_training_estimates(self):
        script = Path(__file__).with_name("dashboard") / "dashboard.js"
        result = subprocess.run(["node", "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({document: {getElementById: () => ({dataset: {}})}, Intl});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const evaluate = (code) => JSON.parse(JSON.stringify(vm.runInContext(code, context)));
assert.deepEqual(evaluate(`sortRuns([{id:'a',name:'rosie-9'},{id:'b',name:'rosie-100'},
    {id:'c',name:'rosie-20'}]).map(run=>run.name)`), ['rosie-100','rosie-20','rosie-9']);
assert.deepEqual(evaluate(`sortRuns([{id:'a',name:'20261018-230000'},{id:'b',name:'20261019-010000'}])
    .map(run=>run.id)`), ['b','a']);
assert.deepEqual(evaluate(`records=[{update:1,loss:2},{update:3,loss:8},{update:4},
    {update:5,loss:10},{update:6,loss:14}]; smoothing=3; smoothSeries('loss',true)`),
    [{x:1,y:null},{x:3,y:1.5},{x:4,y:null},{x:5,y:null},{x:6,y:2}]);
assert.deepEqual(evaluate(`smoothing=1; smoothSeries('loss',true)`),
    [{x:1,y:null},{x:3,y:3},{x:4,y:null},{x:5,y:null},{x:6,y:4}]);
assert.deepEqual(evaluate(`records=[{update:1,reward_types:{death:-10}},
    {update:2,reward_types:{death:-20}},{update:3,reward_types:{death:-5}}];
    smoothing=3; smoothSeries(row=>row.reward_types?.death,true)`),
    [{x:1,y:null},{x:2,y:-5},{x:3,y:5}]);
assert.deepEqual(evaluate(`smoothSeries(row=>row.reward_types?.death)`),
    [{x:1,y:-10},{x:2,y:-15},{x:3,y:-10}]);
assert.equal(evaluate(`definitions.find(([id])=>id==='loss-chart')[1].some(([key])=>key==='policy_loss')`),false);
assert.equal(evaluate(`definitions.find(([id])=>id==='policy-loss-chart')[1][0][0]`),'policy_loss');
assert.equal(evaluate(`definitions.find(([id])=>id==='duration-chart')[1].some(([key])=>key==='evaluation_seconds')`),true);
assert.deepEqual(evaluate(`trainingTime([{update:99,collection_seconds:900,update_seconds:100},
    {update:101,collection_seconds:6,update_seconds:4},{update:102,collection_seconds:8,update_seconds:2}],
    {training:{start_update:100,target_update:110}})`),
    {target:110,remaining:8,average:10,elapsed:20,eta:80});
assert.deepEqual(evaluate(`trainingTime([{update:1,collection_seconds:3,update_seconds:2}],{options:{updates:5}})`),
    {target:5,remaining:4,average:5,elapsed:5,eta:20});
assert.deepEqual(evaluate(`trainingTime([{update:1,collection_seconds:3,update_seconds:2,
    evaluation_seconds:4,replay_seconds:1,iteration_seconds:10}],{options:{updates:5}})`),
    {target:5,remaining:4,average:10,elapsed:10,eta:40});
assert.deepEqual(evaluate(`trainingTime([{update:102,collection_seconds:3,update_seconds:2}],
    {options:{resume:'latest.pt',updates:10}})`),
    {target:null,remaining:null,average:5,elapsed:null,eta:null});
assert.equal(evaluate(`trainingTime([],{training:{start_update:100,target_update:110}}).eta`),null);
assert.equal(evaluate(`trainingTime([{update:111}],{training:{start_update:100,target_update:110}}).eta`),0);
assert.equal(evaluate(`trainingTime([{update:2,completed_curriculum:true}],{options:{updates:1000}}).eta`),0);
assert.equal(evaluate(`trainingTime([{update:1,collection_seconds:3}],{options:{updates:5}}).average`),null);
assert.equal(evaluate(`trainingTime(Array.from({length:25},(_,index)=>({update:index+1,
    collection_seconds:index,update_seconds:1})),{options:{updates:30}}).average`),15.5);
assert.equal(evaluate(`duration(null)`),'--');
assert.equal(evaluate(`duration(90000)`),'1d 1h');
""", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for performance chart checks")
    def test_performance_metrics_and_selection_dom(self):
        script = Path(__file__).with_name("dashboard") / "dashboard.js"
        self.assertNotIn('id="training-aggregation', script.with_name("index.html").read_text())
        result = subprocess.run(["node", "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Node {
    constructor() {this.children=[]; this.dataset={}; this.value='';}
    set textContent(value) {this.text=String(value); this.children=[];}
    get textContent() {return this.text ?? this.children.map(child=>child.textContent).join(' ');}
    set innerHTML(value) {throw new Error('Unsafe HTML');}
    replaceChildren(...children) {this.text=undefined; this.children=children; this.value=children[0]?.value ?? '';}
    append(...children) {this.children.push(...children);}
    addEventListener() {}
    setAttribute(name,value) {this[name]=value;}
    closest() {return {querySelector:()=>({firstChild:{textContent:'Metric'},after(){}})};}
}
const nodes=new Map();
const document={getElementById(id) {if(!nodes.has(id)) nodes.set(id,new Node()); return nodes.get(id);},
    createElement:()=>new Node(),createDocumentFragment:()=>new Node(),querySelector:()=>null,querySelectorAll:()=>[]};
const context=vm.createContext({document,Intl,Option:class extends Node {constructor(label,value){super();this.textContent=label;this.value=value;}}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
context.catalog=JSON.parse(process.argv[2]);
const evaluate=code=>JSON.parse(JSON.stringify(vm.runInContext(code,context)));
vm.runInContext(`metricCatalog=catalog; chartAxis='samples'; windowSize=0; smoothing=10;
records=[
    {update:1,samples:10,stage:'aim',stage_index:0,training_rollout:{training_return:2}},
    {update:2,samples:20,stage:'aim',stage_index:0,training_rollout:{training_return:4},
     evaluations:{combined_benchmark:{snapshot_update:2,win_rate:.75,reward_mean:9,episodes:40}}},
    {update:3,samples:30,stage:'combat',stage_index:1,training_return:-99,training_win_rate:.1,
     training_episodes:1000,training_episode_window:1000,
     training_opponents:{past:{training_win_rate:.05,training_return:-50,training_episodes:20}},
     training_rollout:{training_return:100,training_win_rate:.5,training_episodes:2000,source:'recorded',
         training_opponents:{past:{training_win_rate:.25,training_loss_rate:.75,training_return:-3,training_length:25,training_episodes:20}}},
     evaluations:{combined_benchmark:{snapshot_update:2,win_rate:.75,reward_mean:9,episodes:40}}}
]; renderPerformance();`,context);
assert.equal(document.getElementById('stat-benchmark').textContent,'75%');
assert.match(document.getElementById('stat-benchmark-detail').textContent,/40 episodes/);
assert.match(document.getElementById('stat-benchmark-age').textContent,/1 updates ago/);
assert.equal(document.getElementById('stat-training-win').textContent,'50%');
assert.deepEqual(evaluate("metricPoints('win_rate').map(({x,y})=>({x,y}))"),[{x:20,y:75}]);
assert.deepEqual(evaluate("metricPoints('evaluation_reward').map(point=>point.y)"),[9]);
assert.equal(evaluate("metricPoints('training_return').at(-1).y"),100,'EMA must restart at stage boundary');
assert.equal(evaluate("metricPoints('training_return',true).at(-1).y"),null,'Derivative must restart at stage boundary');
document.getElementById('training-opponent').value='past';
vm.runInContext('renderPerformance();',context);
assert.equal(document.getElementById('stat-training-win').textContent,'25%');
assert.equal(document.getElementById('stat-training-return').textContent,'-3');
assert.deepEqual(evaluate("metricPoints('training_win_rate').map(point=>point.y)"),[null,null,25]);
assert.equal(document.getElementById('metric-select').children.length,Object.keys(context.catalog).length);
assert.equal(document.getElementById('benchmark-performance').hidden,false);
assert.equal(document.getElementById('training-performance').hidden,true);
vm.runInContext(`performanceMode='training';
records.at(-1).training_rollout={training_episodes:2000,training_win_rate:.6,training_return:7,source:'recorded',
    training_opponents:{past:{training_episodes:1200,training_win_rate:.7,training_return:8}}}; renderPerformance();`,context);
assert.equal(document.getElementById('benchmark-performance').hidden,true);
assert.equal(document.getElementById('training-performance').hidden,false);
assert.equal(document.getElementById('reward-diagnostics').hidden,false);
assert.equal(document.getElementById('optimization-title').hidden,undefined);
assert.equal(document.getElementById('runtime-title').hidden,undefined);
assert.equal(document.getElementById('stat-training-win').textContent,'70%');
assert.match(document.getElementById('episode-status').textContent,/1,200 completed episodes in this rollout/);
context.Chart=class {static defaults={font:{}};constructor(canvas,config){this.data=config.data;this.options=config.options;this.plugins=config.plugins;}update(){}};
vm.runInContext('initCharts(); redrawCharts();',context);
assert.equal(evaluate("[...charts.values()].every(chart=>chart.plugins.some(plugin=>plugin.id==='legendSolo'))"),true);
assert.equal(evaluate("charts.get('training-outcomes-chart').data.datasets[0].data.at(-1).y"),70);
assert.equal(evaluate("charts.get('training-outcomes-chart').options.scales.y.max"),100);
assert.equal(evaluate("charts.get('training-outcomes-chart').options.scales.x.title.text"),'Learner transitions');
vm.runInContext('windowSize=1;',context);
assert.deepEqual(evaluate("metricPoints('win_rate')"),[],'Cached evaluations must not move into the latest update window');
vm.runInContext('windowSize=0;',context);
vm.runInContext(`records=[{update:1,reward_mean:.5,training_return:99,training_win_rate:1,
    training_opponents:{past:{training_return:50,training_win_rate:.9}}}]; renderPerformance();`,context);
assert.equal(document.getElementById('stat-training-return').textContent,'--');
assert.equal(document.getElementById('stat-training-win').textContent,'--','Recent history must not substitute for missing rollout data');
assert.match(document.getElementById('episode-status').textContent,/not recorded/);
assert.equal(document.getElementById('stat-benchmark').textContent,'--');
assert.deepEqual(evaluate("metricPoints('training_return').map(point=>point.y)"),[],'Missing samples must not fabricate points');
vm.runInContext(`chartAxis='elapsed_seconds'; records=[{update:1,loss:2},
    {update:2,loss:3,iteration_seconds:10},{update:3,loss:4,elapsed_seconds:50}];`,context);
assert.deepEqual(evaluate("metricPoints('loss').map(({x,y})=>({x,y}))"),[{x:50,y:4}]);
vm.runInContext(`records=[{update:4,samples:40,stage:'combat',training_rollout:{
    training_episodes:100,training_win_rate:.6,training_timeout_rate:.2,training_loss_or_draw_rate:.2,
    training_opponents:{},source:'recovered_rewards'}}]; renderPerformance();`,context);
assert.match(document.getElementById('episode-status').textContent,/Recovered reward counts/);
assert.match(document.getElementById('episode-status').textContent,/loss and draw combined/);
assert.equal(document.getElementById('stat-training-win').textContent,'60%');
assert.equal(document.getElementById('stat-training-return').textContent,'--');
assert.equal(evaluate("metricValue(records[0],'training_loss_rate')"),null);
assert.equal(evaluate("metricValue(records[0],'training_loss_or_draw_rate')"),20);
assert.equal(document.getElementById('benchmark-performance').hidden,true,'Refresh must preserve training mode');
assert.equal(document.getElementById('training-performance').hidden,false);
vm.runInContext(`performanceMode='benchmark'; renderPerformance();`,context);
assert.equal(document.getElementById('benchmark-performance').hidden,false);
assert.equal(document.getElementById('training-performance').hidden,true);
assert.equal(document.getElementById('reward-diagnostics').hidden,true);
assert.equal(evaluate('chartAxis'),'elapsed_seconds');
vm.runInContext(`chartAxis='update'; windowSize=0; records=[{update:1,loss:0},{update:2,loss:100}];`,context);
const slider=document.getElementById('smooth');
const smoothingValues=[0,250,500,750,1000].map(position=>{
    slider.value=String(position);
    vm.runInContext('updateSmoothing();',context);
    const value=evaluate('smoothing');
    assert.equal(document.getElementById('smooth-value').textContent,value===1?'Off':`EMA ${value}`);
    assert.equal(slider['aria-valuetext'],document.getElementById('smooth-value').textContent);
    return value;
});
assert.deepEqual(smoothingValues,[1,5,22,106,500],'Slider scale must increase exponentially from Off to EMA 500');
assert.ok(Math.abs(evaluate("charts.get('loss-chart').data.datasets[1].data.at(-1).y")-200/501)<1e-12,
    'Slider must redraw with the selected EMA');
vm.runInContext('renderPerformance();',context);
assert.equal(evaluate('smoothing'),500,'Refreshing performance must preserve smoothing');
slider.value='0';
vm.runInContext('updateSmoothing();',context);
assert.equal(evaluate("charts.get('loss-chart').data.datasets[1].data.at(-1).y"),100,'Off must restore unsmoothed values');
assert.equal(document.getElementById('smooth-value').textContent,'Off');
vm.runInContext(`smoothing=1; selectedSuite='combined_benchmark';
records=[
    {update:10,samples:100,elapsed_seconds:1,stage:'aim',stage_index:0,loss:0},
    {update:20,samples:200,elapsed_seconds:2,stage:'aim',stage_index:0,loss:10,
        evaluations:{combined_benchmark:{snapshot_update:20,win_rate:.5}}},
    {update:30,samples:300,elapsed_seconds:3,stage:'fight',stage_index:1,loss:30,
        evaluations:{combined_benchmark:{snapshot_update:20,win_rate:.5}}},
    {update:40,samples:400,elapsed_seconds:4,stage:'fight',stage_index:1,loss:50,
        evaluations:{combined_benchmark:{snapshot_update:40,win_rate:.8}}},
    {update:50,samples:500,elapsed_seconds:5,stage:'fight',stage_index:1,loss:70,
        evaluations:{combined_benchmark:{snapshot_update:40,win_rate:.8}}},
    {update:60,samples:600,elapsed_seconds:6,stage:'mixed',stage_index:2,loss:99}
]; windowMode='latest'; windowSize=2;`,context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[50,60]);
vm.runInContext("windowMode='range'; windowStart=20; windowEnd=40;",context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[20,30,40]);
vm.runInContext("windowMode='stage'; windowStage='1';",context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[30,40,50]);
assert.deepEqual(evaluate("metricPoints('win_rate').map(point=>point.update)"),[40],
    'Stage filtering must use benchmark snapshot stages, not rows carrying cached results');
for (const [axis,expected] of [['update',[30,40,50]],['samples',[300,400,500]],['elapsed_seconds',[3,4,5]]]) {
    context.axis=axis;
    vm.runInContext('chartAxis=axis;',context);
    assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.x)"),expected);
}
vm.runInContext("windowMode='range'; windowStart=40; windowEnd=50; smoothing=3;",context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.y)"),[40,55],
    'Cropping must preserve preceding EMA context within the stage');
vm.runInContext("windowMode='stage'; windowStage='3';",context);
assert.deepEqual(evaluate("metricPoints('loss')"),[],'An unrecorded stage must not show another stage');
vm.runInContext("windowMode='all';",context);
assert.equal(evaluate("metricPoints('loss').length"),6);
vm.runInContext(`configuration={experiment:{stages:[{name:'aim'},{name:'fight'},{name:'mixed'},{name:'final'}]}};
    chartAxis='update'; smoothing=1; windowSize=2; updateWindowMode('latest');`,context);
assert.equal(document.getElementById('window-latest-control').hidden,false);
assert.equal(document.getElementById('window-range-control').hidden,true);
assert.equal(document.getElementById('window-size-value').textContent,'2 updates');
assert.equal(document.getElementById('window-size')['aria-valuetext'],'2 updates');
for (const [position,expected] of [[0,1],[500,2],[1000,6]]) {
    document.getElementById('window-size').value=String(position);
    vm.runInContext('updateLatestWindow();',context);
    assert.equal(evaluate('windowSize'),expected);
}
vm.runInContext(`windowStart=windowEnd=null; updateWindowMode('range');
    updateWindowRange('start','25'); updateWindowRange('end','45');`,context);
assert.equal(document.getElementById('window-range-control').hidden,false);
assert.equal(document.getElementById('window-latest-control').hidden,true);
assert.equal(Number(document.getElementById('window-start').value),25);
assert.equal(Number(document.getElementById('window-start-slider').value),25);
assert.equal(Number(document.getElementById('window-end').value),45);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[30,40]);
assert.match(document.getElementById('window-info').textContent,/2 of 6 updates.*Updates 30-40/);
vm.runInContext("updateWindowRange('end','');",context);
assert.equal(evaluate('windowEnd'),45,'An unfinished numeric edit must not become update zero');
vm.runInContext("updateWindowRange('start','10000');",context);
assert.deepEqual(evaluate('[windowStart,windowEnd]'),[60,60],'Crossing handles must keep bounds ordered and clamp to the run');
vm.runInContext("updateWindowRange('end','-10');",context);
assert.deepEqual(evaluate('[windowStart,windowEnd]'),[10,10]);
vm.runInContext("updateWindowRange('end','45'); updateWindowRange('start','25'); redrawCharts();",context);
assert.deepEqual(evaluate('[windowStart,windowEnd]'),[25,45],'Refresh must preserve explicit endpoints');
vm.runInContext("windowStage=''; updateWindowMode('stage');",context);
assert.equal(evaluate('windowStage'),'2','First stage selection should use the latest recorded stage');
assert.equal(document.getElementById('window-stage-control').hidden,false);
assert.match(document.getElementById('window-stage').children[1].textContent,/fight.*updates 30-50/);
document.getElementById('window-stage').value='3';
vm.runInContext("windowStage='3'; redrawCharts();",context);
assert.deepEqual(evaluate("metricPoints('loss')"),[],'A configured future stage must remain empty');
assert.match(document.getElementById('window-info').textContent,/0 of 6 updates/);
document.getElementById('window-stage').value='2';
vm.runInContext("windowStage='2'; records.push({update:70,samples:700,elapsed_seconds:7,stage:'mixed',stage_index:2,loss:110}); redrawCharts();",context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[60,70],'Stage windows must include new updates in that stage');
vm.runInContext("records.push({update:80,samples:800,elapsed_seconds:8,stage:'final',stage_index:3,loss:120}); redrawCharts();",context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[60,70],'Stage windows must not follow training into the next stage');
vm.runInContext("chartAxis='samples'; redrawCharts();",context);
assert.equal(evaluate("charts.get('loss-chart').options.scales.x.min"),600);
assert.equal(evaluate("charts.get('loss-chart').options.scales.x.max"),700);
vm.runInContext("updateWindowMode('range');",context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[30,40],'Range must remain fixed as training grows');
vm.runInContext("windowSize=2; updateWindowMode('latest');",context);
assert.deepEqual(evaluate("metricPoints('loss').map(point=>point.update)"),[70,80]);
vm.runInContext("updateWindowMode('all');",context);
assert.equal(evaluate("metricPoints('loss').length"),8);
vm.runInContext("configuration=null; chartAxis='update'; records=[{update:1,stage:'legacy',loss:2}]; windowStage=''; updateWindowMode('stage');",context);
assert.equal(evaluate('windowStage'),'legacy','Historical stage names must work without indexed config');
assert.equal(evaluate("metricPoints('loss').length"),1);
vm.runInContext("records=[]; redrawCharts();",context);
assert.equal(document.getElementById('window-size').disabled,true);
assert.equal(document.getElementById('window-start').disabled,true);
assert.equal(document.getElementById('window-stage').disabled,true);
assert.match(document.getElementById('window-info').textContent,/0 of 0 updates/);
""", str(script), json.dumps(CATALOG)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_episode_metrics_are_sanitized_and_catalog_is_shared(self):
        row = {"update": 1, "training_win_rate": .25, "training_return": -2, "training_episodes": 8,
               "training_opponents": {"<past>": {"training_win_rate": .5, "training_return": -3,
                                                   "training_episodes": 2, "ignored": "unused"}}}
        invalid = {**row, "update": 2, "training_opponents": {"past": {"training_win_rate": 25}}}
        self.metrics.write_text(json.dumps(row) + "\n" + json.dumps(invalid) + "\n")
        result = self.store.read(self.store.list_runs()[0]["id"])
        self.assertEqual(result["metric_catalog"], CATALOG)
        self.assertEqual(len(result["metrics"]), 1)
        opponent = result["metrics"][0]["training_opponents"]["<past>"]
        self.assertEqual(opponent["training_win_rate"], .5)
        self.assertNotIn("ignored", opponent)
        self.assertIn("Skipped 1", result["warnings"][0])

    def test_historical_rollout_recovery_is_conservative_and_read_only(self):
        config = {"metadata": {"learner": "opponent_pool"}, "experiment": {"stages": [
            {"name": "combat", "rewards": {"weights": {"won_duel": 20, "death": -20}}}]}}
        (self.run_dir / "config.json").write_text(json.dumps(config))
        row = {"update": 1, "stage_index": 0, "stage": "combat", "terminated_duels": 8, "timeouts": 2,
               "reward_types": {"won_duel": 120, "death": -40}}
        self.metrics.write_text(json.dumps(row) + "\n")
        before = self.metrics.read_bytes()
        result = self.store.read(self.store.list_runs()[0]["id"])["metrics"][0]["training_rollout"]
        self.assertEqual(result["training_episodes"], 10)
        self.assertEqual(result["training_win_rate"], .6)
        self.assertEqual(result["training_timeout_rate"], .2)
        self.assertEqual(result["training_loss_or_draw_rate"], .2)
        self.assertEqual(result["source"], "recovered_rewards")
        for field in ("training_return", "training_length", "training_loss_rate", "training_draw_rate"):
            self.assertNotIn(field, result)
        self.assertEqual(self.metrics.read_bytes(), before)
        for changes in ({"reward_types": {"won_duel": 123, "death": -40}}, {"terminated_duels": 9},
                        {"stage": "other"}, {"stage_index": 5}):
            self.metrics.write_text(json.dumps({**row, **changes}) + "\n")
            self.assertNotIn("training_rollout", self.store.read(self.store.list_runs()[0]["id"])["metrics"][0])
        config["experiment"]["stages"][0]["rewards"]["weights"]["won_duel"] = 0
        (self.run_dir / "config.json").write_text(json.dumps(config))
        self.metrics.write_text(json.dumps(row) + "\n")
        self.assertNotIn("training_rollout", self.store.read(self.store.list_runs()[0]["id"])["metrics"][0])

    def test_recorded_rollout_takes_precedence_and_is_sanitized(self):
        row = {"update": 1, "training_rollout": {"training_episodes": 2000, "training_win_rate": .75,
               "training_return": -3, "training_opponents": {"past": {"training_episodes": 100,
               "training_win_rate": .5}}, "source": "recorded"}}
        self.metrics.write_text(json.dumps(row) + "\n")
        result = self.store.read(self.store.list_runs()[0]["id"])["metrics"][0]
        self.assertEqual(result["training_rollout"], row["training_rollout"])
        row["training_rollout"]["training_opponents"]["past"]["training_win_rate"] = 50
        self.metrics.write_text(json.dumps(row) + "\n")
        self.assertEqual(self.store.read(self.store.list_runs()[0]["id"])["metrics"], [])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".dashboard-tests-", dir=".")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.run_dir = self.root / "runs" / "first"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "config.json").write_text('{"options":{"device":"cpu","envs":4}}', encoding="utf-8")
        self.metrics = self.run_dir / "metrics.jsonl"
        self.metrics.write_bytes(b'{"update":1,"reward_mean":0.25,"samples":16}\n')
        self.store = RunStore(self.root / "runs")

    def test_fixture_preserves_unittest_runner(self):
        self.assertTrue(callable(self.run))

    def test_discovery_custom_output_and_missing_root(self):
        self.assertEqual(len(self.store.list_runs()), 1)
        identity = self.store.list_runs()[0]["id"]
        self.assertEqual(self.store.read(identity)["config"]["options"]["envs"], 4)
        self.assertEqual(RunStore(self.root / "missing").list_runs(), [])
        custom = RunStore(self.root / "missing", [self.run_dir, self.run_dir])
        self.assertEqual(custom.list_runs()[0]["id"], identity)
        self.assertEqual(len(custom.list_runs()), 1)
        with self.assertRaises(FileNotFoundError):
            self.store.read("../../config.json")

    def test_partial_tail_and_live_append(self):
        identity = self.store.list_runs()[0]["id"]
        with self.metrics.open("ab") as output:
            output.write(b'{"update":2,"reward_mean":')
        self.assertEqual(len(self.store.read(identity)["metrics"]), 1)
        with self.metrics.open("ab") as output:
            output.write(b'0.5}\n{"update":3}\n')
        self.assertEqual([row["update"] for row in self.store.read(identity)["metrics"]], [1, 2, 3])
        self.metrics.write_bytes(b'{"update":1,"reward_mean":-1}\n')
        self.assertEqual(self.store.read(identity)["metrics"][0]["reward_mean"], -1)

    def test_run_summaries_are_searchable_cached_and_read_only(self):
        config_path = self.run_dir / "config.json"
        config = {"options": {"seed": 7, "lr": 0.1},
                  "metadata": {"schema": {"id": "equipment"}, "environment": {"name": "Armored duel"}},
                  "experiment": {"name": "Width experiment", "seed": 42, "learner": {"hidden_dim": 512, "lr": 0.0003}}}
        original = json.dumps(config).encode()
        config_path.write_bytes(original)
        metrics = self.metrics.read_bytes()
        with mock.patch.object(self.store, "read_config", wraps=self.store.read_config) as read:
            first = self.store.list_runs()[0]
            self.assertEqual(first["experiment_name"], "Width experiment")
            self.assertEqual(first["environment_name"], "Armored duel")
            self.assertEqual(first["policy_name"], "equipment")
            self.assertEqual(first["seed"], 42)
            self.assertEqual(first["hidden_dim"], 512)
            self.assertEqual(first["lr"], 0.0003)
            self.assertEqual(first["warnings"], [])
            self.assertEqual(self.store.list_runs()[0], first)
            self.assertEqual(read.call_count, 1, "Polling must reuse unchanged configuration summaries")
            config["experiment"]["name"] = "Updated experiment name"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(self.store.list_runs()[0]["experiment_name"], "Updated experiment name")
            self.assertEqual(read.call_count, 2)
        self.assertEqual(self.metrics.read_bytes(), metrics)
        self.assertEqual(json.loads(config_path.read_bytes()), config, "Discovery must never write training files")
        config_path.write_text('{"options":', encoding="utf-8")
        self.assertIn("Configuration is not valid JSON yet.", self.store.list_runs()[0]["warnings"])
        for malformed in ({"metadata": []}, {"experiment": None}, {"options": {"lr": float("nan")}}):
            config_path.write_text(json.dumps(malformed), encoding="utf-8")
            self.assertTrue(self.store.list_runs()[0]["warnings"], "Bad metadata must be visible, not silently hidden")
        config_path.unlink()
        self.assertEqual(self.store.list_runs()[0]["warnings"], [])
        self.assertNotIn("experiment_name", self.store.list_runs()[0])

    def test_run_order_uses_activity_dates_not_names(self):
        for path in (self.metrics, self.run_dir / "config.json"):
            os.utime(path, (100, 100))
        newer = self.root / "runs" / "aaa"
        newer.mkdir()
        config = newer / "config.json"
        config.write_text("{}", encoding="utf-8")
        os.utime(config, (200, 200))
        runs = self.store.list_runs()
        self.assertEqual([run["name"] for run in runs], ["aaa", "first"])
        self.assertEqual([run["modified"] for run in runs], [200, 100])

    def test_invalid_records_and_config_only_run(self):
        identity = self.store.list_runs()[0]["id"]
        with self.metrics.open("ab") as output:
            output.write(b'bad\n[]\n{"update":true}\n{"update":2,"loss":NaN}\n{"update":3}\n')
        result = self.store.read(identity)
        self.assertEqual([row["update"] for row in result["metrics"]], [1, 3])
        self.assertIn("Skipped 4", result["warnings"][0])
        (self.run_dir / "config.json").write_text('{"options":', encoding="utf-8")
        self.assertIsNone(self.store.read(identity)["config"])
        self.metrics.unlink()
        self.assertEqual(self.store.read(identity)["metrics"], [])

    def test_resume_replaces_updates_after_an_older_checkpoint(self):
        with self.metrics.open("ab") as output:
            output.write(b'{"update":2,"samples":32}\n{"update":3,"samples":48}\n')
            output.write(b'{"update":2,"samples":64}\n')
        identity = self.store.list_runs()[0]["id"]
        rows = self.store.read(identity)["metrics"]
        self.assertEqual([row["update"] for row in rows], [1, 2])
        self.assertEqual(rows[-1]["samples"], 64)

    def test_resolved_experiment_and_metric_metadata(self):
        experiment = {"format": "minecraft_sim_experiment", "version": 1, "name": "combat", "seed": 17,
                      "policy": {"observations": [], "actions": []},
                      "learner": {"hidden_dim": 32, "lr": 0.001, "epochs": 2, "minibatch_size": 8},
                      "opponents": {"target": {"type": "scripted", "controller": "stationary_target"}},
                      "stages": [{"name": "aim", "environment_name": "duel", "arena_names": ["flat"],
                                  "scenario": {"arenas": [{"environment": {"version": 1, "name": "flat"}, "weight": 1}]},
                                  "controls": {"attack": False}, "advance": {"updates": 2},
                                  "rewards": {"weights": {"doing_damage": 1}},
                                  "opponents": [{"id": "target", "weight": 1}]}],
                      "evaluations": {"held": {"episodes": 4}}}
        config = {"options": {"device": "cpu"}, "metadata": {"learner": "opponent_pool",
                  "environment": {"version": 1, "name": "deployment"}, "controls": {"attack": False},
                  "max_ticks": 50, "hidden_dim": 32, "reward_weights": {"doing_damage": 1}},
                  "experiment": experiment}
        (self.run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        row = {"update": 2, "stage": "aim", "stage_index": 0, "next_stage_index": 1,
               "stage_progress": {"updates": 2, "episodes": 4, "transitions": 128}, "completed_curriculum": False,
               "evaluations": {"held": {"win_rate": 0.75, "reward_mean": -1.5, "episodes": 4}},
               "curriculum_boundary": {"reset_worlds": 2, "next_stage": "combat"}}
        self.metrics.write_text(json.dumps(row) + "\n", encoding="utf-8")
        identity = self.store.list_runs()[0]["id"]
        data = self.store.read(identity)
        self.assertEqual(data["config"], config)
        self.assertEqual(data["metrics"], [row])
        self.assertEqual(data["warnings"], [])
        row.update(evaluation_seconds=0.125, evaluation_episodes=4, replay_seconds=0.25,
               iteration_seconds=1.0, cuda_peak_allocated_bytes=1024, cuda_peak_reserved_bytes=2048)
        row["evaluations"]["held"].update(worlds=4, seconds=0.125, snapshot_update=2, stage="aim", stage_index=0)
        row.update(update=3, next_stage_index=None, completed_curriculum=True)
        row["curriculum_boundary"]["next_stage"] = None
        with self.metrics.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")
        self.assertEqual(self.store.read(identity)["metrics"][-1], row)
        config["plan"] = config.pop("experiment")
        (self.run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.assertEqual(self.store.read(identity)["config"], config)

    def test_invalid_research_metadata_is_skipped_without_losing_valid_rows(self):
        invalid = [
            {"stage": 1}, {"stage": ""}, {"stage_index": True}, {"stage_index": -1},
            {"next_stage_index": 1.5}, {"completed_curriculum": 1},
            {"stage_progress": []}, {"stage_progress": {"updates": 1}},
            *({"stage_progress": {"updates": 1, "episodes": value, "transitions": 4}}
              for value in (True, -1, 1.5, "2", None)),
            {"evaluations": []}, {"evaluations": {"held": None}}, {"evaluations": {"held": {}}},
            *({"evaluations": {"held": {"win_rate": value, "reward_mean": 0, "episodes": 2}}}
              for value in (float("nan"), float("inf"), -0.1, 1.1, True, None)),
            {"evaluations": {"held": {"win_rate": 0.5, "reward_mean": float("inf"), "episodes": 2}}},
            {"evaluations": {"held": {"win_rate": 0.5, "reward_mean": 1, "episodes": 2.5}}},
                        *({"evaluations": {"held": {"win_rate": 0.5, "reward_mean": 1, "episodes": 2, **extra}}}
                            for extra in ({"worlds": True}, {"seconds": -1}, {"seconds": float("inf")},
                                                        {"snapshot_update": -1}, {"stage_index": "0"}, {"stage": ""})),
            {"evaluations": {"": {"win_rate": 0, "reward_mean": 0, "episodes": 0}}},
            {"curriculum_boundary": []}, {"curriculum_boundary": {"reset_worlds": 2}},
            {"curriculum_boundary": {"reset_worlds": True, "next_stage": "combat"}},
            {"curriculum_boundary": {"reset_worlds": 2, "next_stage": {}}},
            {"samples": 1.5}, {"timeouts": -1}, {"completed_rounds": False},
            {"evaluation_episodes": True}, {"cuda_peak_allocated_bytes": -1},
        ]
        with self.metrics.open("a", encoding="utf-8") as output:
            for index, fields in enumerate(invalid, 2):
                output.write(json.dumps({"update": index, **fields}) + "\n")
            output.write(json.dumps({"update": len(invalid) + 2, "completed_curriculum": True,
                                     "evaluations": {"held": {"win_rate": 1, "reward_mean": 0, "episodes": 0}}}) + "\n")
        identity = self.store.list_runs()[0]["id"]
        data = self.store.read(identity)
        self.assertEqual([row["update"] for row in data["metrics"]], [1, len(invalid) + 2])
        self.assertEqual(data["warnings"], [f"Skipped {len(invalid)} invalid metric record(s)."])

    def test_reward_history_and_logged_contributions(self):
        self.metrics.write_text(
            '{"update":1,"samples":16,"reward_mean":0.25}\n'
            '{"update":2,"samples":48,"reward_mean":-0.5}\n'
            '{"update":3,"samples":64,"reward_mean":0.5,"reward_total":7.5,'
            '"reward_types":{"doing_damage":10,"taking_damage":-2.5}}\n', encoding="utf-8")
        identity = self.store.list_runs()[0]["id"]
        rows = self.store.read(identity)["metrics"]
        self.assertEqual([row["reward_total"] for row in rows], [4, -16, 7.5])
        self.assertEqual(rows[-1]["reward_types"], {"doing_damage": 10, "taking_damage": -2.5})
        self.metrics.write_text('{"update":10,"samples":160,"reward_mean":1}\n'
                                '{"update":12,"samples":192,"reward_mean":1}\n', encoding="utf-8")
        self.assertTrue(all("reward_total" not in row for row in self.store.read(identity)["metrics"]))
        self.metrics.write_text('{"update":1,"reward_types":{"death":NaN}}\n', encoding="utf-8")
        self.assertEqual(self.store.read(identity)["metrics"], [])

    def write_replay(self, directory=None):
        directory = directory or self.run_dir / "replays"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "training-round-000000010.jsonl"
        header = {"format": "minecraft_sim_replay", "version": 3, "players": 2,
                  "ticks_per_second": 20, "environment": {"players": []}}
        frames = [{"tick": tick, "actors": [
            {"id": index, "position": [index + tick, 1, 0], "yaw": 0, "pitch": 0, "health": 20}
            for index in (1, 0)]} for tick in range(3)]
        path.write_text("".join(json.dumps(value) + "\n" for value in [header, *frames]), encoding="utf-8")
        return path

    def test_replay_discovery_and_validation(self):
        identity = self.store.list_runs()[0]["id"]
        self.assertEqual(self.store.list_replays(identity), [])
        path = self.write_replay()
        path.with_suffix(".jsonl.part").write_text("partial", encoding="utf-8")
        (path.parent / "latest.pt").write_bytes(b"not a replay")
        (path.parent / "link.jsonl").symlink_to(path)
        replays = self.store.list_replays(identity)
        self.assertEqual([row["name"] for row in replays], [path.name])
        data = self.store.read_replay(identity, replays[0]["id"])
        self.assertEqual([frame["tick"] for frame in data["frames"]], [0, 1, 2])
        self.assertEqual([actor["id"] for actor in data["frames"][0]["actors"]], [0, 1])
        with self.assertRaises(FileNotFoundError):
            self.store.read_replay(identity, "../latest.pt")
        with self.assertRaises(FileNotFoundError):
            self.store.list_replays("unknown")
        original = path.read_bytes()
        for content in (original.rstrip(), original.replace(b'"tick": 1', b'"tick": 0'),
                        original.replace(b'"health": 20', b'"health": NaN'),
                        original.replace(b'"version": 3', b'"version": 99'),
                        original.replace(b'"ticks_per_second": 20', b'"ticks_per_second": 0'),
                        original.replace(b'"id": 1', b'"id": 0')):
            path.write_bytes(content)
            with self.subTest(content=content[:100]), self.assertRaises(ValueError):
                self.store.read_replay(identity, replays[0]["id"])
        path.write_bytes(original)
        with mock.patch("simulator.training.dashboard.MAX_REPLAY_BYTES", 10):
            self.assertEqual(self.store.list_replays(identity), [])
            with self.assertRaises(ValueError):
                self.store.read_replay(identity, replays[0]["id"])

    def test_configured_replay_directory(self):
        path = self.write_replay(self.root / "exports")
        (self.run_dir / "config.json").write_text(json.dumps({"options": {"replay_dir": str(path.parent)}}), encoding="utf-8")
        identity = self.store.list_runs()[0]["id"]
        self.assertEqual(self.store.list_replays(identity)[0]["name"], path.name)

    def test_dashboard_replay_request_lifecycle(self):
        identity = self.store.list_runs()[0]["id"]
        with self.assertRaises(ReplayRequestUnsupported):
            self.store.request_replay(identity)
        (self.run_dir / "config.json").write_text(
            json.dumps({"options": {"dashboard_replay_requests": True}}), encoding="utf-8")
        self.assertEqual(self.store.replay_request(identity), {"supported": True, "request": None})
        request = self.store.request_replay(identity)
        self.assertEqual(request["status"], "pending")
        with self.assertRaises(ReplayRequestConflict):
            self.store.request_replay(identity)
        self.assertIsNone(requested_replay_round(100, 100, []))
        self.assertEqual(requested_replay_round(100, 101, []), 101)
        self.assertEqual(requested_replay_round(100, 101, [100, 101]), 100)
        self.assertTrue(finish_request(self.run_dir, request["request_id"], "completed",
                                       replay="research-000000101.jsonl", completed_rounds=101))
        status = self.store.replay_request(identity)
        self.assertEqual(status["request"]["status"], "completed")
        self.assertEqual(status["request"]["replay"], "research-000000101.jsonl")
        self.assertNotEqual(self.store.request_replay(identity)["request_id"], request["request_id"])

    def test_http_replay_request_routes(self):
        self.write_replay()
        (self.run_dir / "config.json").write_text(
            json.dumps({"options": {"dashboard_replay_requests": True}}), encoding="utf-8")
        with make_server(self.store, 0) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urlopen(base + "/api/runs", timeout=3) as response:
                    identity = json.load(response)["runs"][0]["id"]
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                with urlopen(base + f"/api/runs/{identity}/metrics", timeout=3) as response:
                    self.assertEqual(json.load(response)["metrics"][0]["samples"], 16)
                with urlopen(base + f"/api/runs/{identity}/replays", timeout=3) as response:
                    replay_id = json.load(response)["replays"][0]["id"]
                with urlopen(base + f"/api/runs/{identity}/replays/{replay_id}", timeout=3) as response:
                    self.assertEqual(len(json.load(response)["frames"]), 3)
                with urlopen(base + f"/api/runs/{identity}/replay-request", timeout=3) as response:
                    status = json.load(response)
                    self.assertTrue(status["supported"])
                    self.assertIsNone(status["request"])
                request_url = base + f"/api/runs/{identity}/replay-request"
                with urlopen(Request(request_url, data=b"", method="POST"), timeout=3) as response:
                    self.assertEqual(response.status, 202)
                    self.assertEqual(json.load(response)["request"]["status"], "pending")
                for route, content_type in (("/", "text/html"), ("/dashboard.css", "text/css"),
                                            ("/dashboard.js", "text/javascript"),
                                            ("/arena.js", "text/javascript"),
                                            ("/arena-data.js", "text/javascript"),
                                            ("/three.module.min.js", "text/javascript"),
                                            ("/three.core.min.js", "text/javascript"),
                                            ("/OrbitControls.js", "text/javascript"),
                                            ("/lucide.min.js", "text/javascript"),
                                            ("/chart.umd.min.js", "text/javascript")):
                    with urlopen(base + route, timeout=3) as response:
                        self.assertEqual(response.headers.get_content_type(), content_type)
                        self.assertGreater(len(response.read()), 100)
                for request, status in ((base + "/api/runs/unknown/metrics", 404),
                                        (base + f"/api/runs/{identity}/replays/unknown", 404),
                                        (base + "/../config.json", 404),
                                        (Request(request_url, data=b"", method="POST"), 409),
                                        (Request(base + "/api/runs", b"{}"), 405)):
                    with self.assertRaises(HTTPError) as failure:
                        urlopen(request, timeout=3)
                    self.assertEqual(failure.exception.code, status)
                    failure.exception.close()
            finally:
                server.shutdown()
                worker.join(timeout=3)


if __name__ == "__main__":
    unittest.main()