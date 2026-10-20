import json
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


class DashboardChecks(unittest.TestCase):
    def test_page_sections_prioritize_run_overview(self):
        class PageStructure(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
                self.links = []
                self.disclosures = []

            def handle_starttag(self, tag, attrs):
                attributes = dict(attrs)
                if "id" in attributes:
                    self.ids.append(attributes["id"])
                if tag == "a":
                    self.links.append(attributes.get("href", ""))
                if tag == "details":
                    self.disclosures.append(attributes)

        page = PageStructure()
        page.feed(Path(__file__).with_name("dashboard").joinpath("index.html").read_text(encoding="utf-8"))
        self.assertEqual(len(page.ids), len(set(page.ids)), "Dashboard IDs must remain unique")
        ordered = ["run-title", "run-identity", "run-context", "stat-update", "stat-remaining",
                   "results", "outcomes-title", "optimization-title", "runtime-title", "curriculum",
                   "world", "run-settings", "recent-updates"]
        positions = [page.ids.index(identity) for identity in ordered]
        self.assertEqual(positions, sorted(positions), "Run identity and metrics must precede detailed sections")
        for target in ("results", "curriculum", "world", "run-settings", "recent-updates"):
            self.assertIn(f"#{target}", page.links)
        self.assertTrue(page.disclosures)
        self.assertTrue(all("open" not in attrs for attrs in page.disclosures), "Detailed configuration starts collapsed")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for curriculum DOM checks")
    def test_experiment_stage_evaluation_and_replay_dom(self):
        directory = Path(__file__).with_name("dashboard")
        result = subprocess.run([shutil.which("node"), "-e", r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
class Node {
    constructor() {this.children=[]; this.dataset={}; this.attributes={}; this.hidden=false; this.value='run';}
    set textContent(value) {this.text=String(value); this.children=[];}
    get textContent() {return this.text ?? this.children.map(child=>child.textContent).join(' ');}
    set innerHTML(value) {throw new Error('Untrusted content must never use innerHTML');}
    append(...children) {this.text=undefined; this.children.push(...children);}
    replaceChildren(...children) {this.text=undefined; this.children=children;}
    addEventListener() {}
    setAttribute(name,value) {this.attributes[name]=value;}
}
const nodes = new Map();
const document = {getElementById(id) {if (!nodes.has(id)) nodes.set(id,new Node()); return nodes.get(id);},
    createElement:()=>new Node(), createDocumentFragment:()=>new Node(), querySelectorAll:()=>[],
    addEventListener() {}, dispatchEvent() {}};
const context = vm.createContext({document, Intl, CustomEvent:class {}, setInterval() {}, cancelAnimationFrame() {}});
vm.runInContext(fs.readFileSync(path.join(process.argv[1],'dashboard.js'),'utf8'), context);
const text = id=>document.getElementById(id).textContent;
const hidden = id=>document.getElementById(id).hidden;
const scene = name=>({name,max_ticks:50,players:[],bounds:{min:[0,0,0],size:[8,4,8]}});
const experiment = {format:'minecraft_sim_experiment',version:1,name:'<img src=x onerror=alert(1)>',seed:17,
    learner:{lr:0.01,epochs:3,minibatch_size:16},policy:{},evaluations:{'<held>':{},unseen:{}},
    stages:[
        {name:'aim',environment_name:'Aim environment',arena_names:['Flat'],scenario:{arenas:[{environment:scene('flat'),weight:1}]},
            controls:{attack:false},opponents:[{id:'target',weight:1}],advance:{updates:2}},
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
assert.equal(text('milestone-position'),'Stage 1 of 2');
assert.match(text('milestone-current'),/Initial stage: aim/);
assert.equal(document.getElementById('milestone-previous').disabled,true);
const milestoneSlots = () => document.getElementById('milestone-strip').children[0].children;
assert.equal(milestoneSlots().length,3);
assert.equal(milestoneSlots()[1].attributes['aria-pressed'],'true','The selected stage must occupy the center slot');
assert.equal(milestoneSlots()[1].attributes['aria-current'],'step');
document.getElementById('milestone-next').onclick();
assert.equal(text('milestone-name'),'fight');
assert.equal(text('milestone-position'),'Stage 2 of 2');
assert.equal(milestoneSlots()[1].attributes['aria-pressed'],'true');
assert.equal(milestoneSlots()[0].attributes['aria-current'],'step','Current and selected stage are separate states');
assert.match(text('milestone-environment'),/Fight environment.*Hill.*Pit/);
assert.match(text('milestone-opponents'),/sword.*25%.*axe.*75%/);
assert.match(text('milestone-observation'),/No rollout recorded/);
assert.equal(document.getElementById('milestone-next').disabled,true);
assert.match(text('milestone-current'),/Initial stage: aim/,'Browsing must not change the current stage');
vm.runInContext('renderCurriculum();',context);
assert.equal(text('milestone-name'),'fight','Live refresh must preserve the browsed stage');
document.getElementById('milestone-return').onclick();
assert.equal(text('milestone-name'),'aim');
assert.match(text('stage-context'),/Aim environment.*Flat.*target.*100%.*attack=false/);
assert.match(text('config'),/Seed 17.*Learning rate 0.01/);
assert.equal(hidden('experiment-details'),false);
assert.equal(document.getElementById('experiment-details').open,undefined,'Research details must default collapsed');
assert.equal(text('curriculum-title'),experiment.name,'Labels must be literal text');
assert.deepEqual(JSON.parse(text('experiment-json')),experiment);
assert.match(text('environment-name'),/Deployment snapshot/);
assert.match(text('evaluations'),/<held>.*--.*unseen/);
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
        result = subprocess.run([shutil.which("node"), "-e", r"""
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
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", r"""
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
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", r"""
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
const scenario = {players:[{loadout:{weapons:['minecraft:iron_sword','minecraft:netherite_axe'],
    selected:0,offhand:'minecraft:shield',armor:{helmet:'minecraft:diamond_helmet'}}}]};
const [player] = replayPlayers(scenario, [{...frames[0].actors[0],weapon:'minecraft:netherite_axe',shield_equipped:false}]);
assert.equal(player.loadout.selected, 1);
assert.equal(player.loadout.offhand, 'minecraft:air');
assert.equal(player.loadout.armor.helmet, 'minecraft:diamond_helmet');
assert.equal(replayPlayers(scenario,[frames[0].actors[0]])[0].loadout.selected,0);
""", str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for dashboard chart checks")
    def test_chart_math_and_training_estimates(self):
        script = Path(__file__).with_name("dashboard") / "dashboard.js"
        result = subprocess.run([shutil.which("node"), "-e", r"""
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
assert.deepEqual(evaluate(`trainingTime([{update:99,collection_seconds:900,update_seconds:100},
    {update:101,collection_seconds:6,update_seconds:4},{update:102,collection_seconds:8,update_seconds:2}],
    {training:{start_update:100,target_update:110}})`),
    {target:110,remaining:8,average:10,elapsed:20,eta:80});
assert.deepEqual(evaluate(`trainingTime([{update:1,collection_seconds:3,update_seconds:2}],{options:{updates:5}})`),
    {target:5,remaining:4,average:5,elapsed:5,eta:20});
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

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".dashboard-tests-", dir=".")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.run = self.root / "runs" / "first"
        self.run.mkdir(parents=True)
        (self.run / "config.json").write_text('{"options":{"device":"cpu","envs":4}}', encoding="utf-8")
        self.metrics = self.run / "metrics.jsonl"
        self.metrics.write_bytes(b'{"update":1,"reward_mean":0.25,"samples":16}\n')
        self.store = RunStore(self.root / "runs")

    def test_discovery_custom_output_and_missing_root(self):
        self.assertEqual(len(self.store.list_runs()), 1)
        identity = self.store.list_runs()[0]["id"]
        self.assertEqual(self.store.read(identity)["config"]["options"]["envs"], 4)
        self.assertEqual(RunStore(self.root / "missing").list_runs(), [])
        custom = RunStore(self.root / "missing", [self.run, self.run])
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

    def test_invalid_records_and_config_only_run(self):
        identity = self.store.list_runs()[0]["id"]
        with self.metrics.open("ab") as output:
            output.write(b'bad\n[]\n{"update":true}\n{"update":2,"loss":NaN}\n{"update":3}\n')
        result = self.store.read(identity)
        self.assertEqual([row["update"] for row in result["metrics"]], [1, 3])
        self.assertIn("Skipped 4", result["warnings"][0])
        (self.run / "config.json").write_text('{"options":', encoding="utf-8")
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
        (self.run / "config.json").write_text(json.dumps(config), encoding="utf-8")
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
        row.update(update=3, next_stage_index=None, completed_curriculum=True)
        row["curriculum_boundary"]["next_stage"] = None
        with self.metrics.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")
        self.assertEqual(self.store.read(identity)["metrics"][-1], row)
        config["plan"] = config.pop("experiment")
        (self.run / "config.json").write_text(json.dumps(config), encoding="utf-8")
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
            {"evaluations": {"": {"win_rate": 0, "reward_mean": 0, "episodes": 0}}},
            {"curriculum_boundary": []}, {"curriculum_boundary": {"reset_worlds": 2}},
            {"curriculum_boundary": {"reset_worlds": True, "next_stage": "combat"}},
            {"curriculum_boundary": {"reset_worlds": 2, "next_stage": {}}},
            {"samples": 1.5}, {"timeouts": -1}, {"completed_rounds": False},
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
        directory = directory or self.run / "replays"
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
        (self.run / "config.json").write_text(json.dumps({"options": {"replay_dir": str(path.parent)}}), encoding="utf-8")
        identity = self.store.list_runs()[0]["id"]
        self.assertEqual(self.store.list_replays(identity)[0]["name"], path.name)

    def test_http_read_only_routes(self):
        self.write_replay()
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