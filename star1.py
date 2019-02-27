import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY
from sc2.constants import VOIDRAY, CYBERNETICSCORE, STALKER, STARGATE
from sc2.constants import ROBOTICSFACILITY, OBSERVER
import random
import cv2
import numpy as np
import time
import keras
# cd C:\Anaconda3\Scripts
# activate missile
# cd C:\Users\Tiny Ants\AppData\Roaming\Python\Python36\site-packages\sc2
HEADLESS = False


class tiny(sc2.BotAI):
    # 165 frame = 1 minute
    def __init__(self, use_model=False):
        self.ITERATIONS_PER_MINUTE = 165
        self.do_something_after = 0
        self.train_data = []
        self.MAX_WORKERS = 55
        self.use_model = use_model
        if self.use_model:
            print("using model")
            self.model = keras.models.load_model("BasicCNN-30-epochs-0.0001-LR-4.2")

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            np.save("train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data))
            print('--- game saving ---')

    async def on_step(self, iteration):
        self.iteration = iteration
        await self.distribute_workers()  # in sc2/bot_ai.py
        await self.build_probe()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        await self.build_army_buildings()
        await self.build_army()
        await self.attack()
        await self.intel()
        await self.scout()

    def random_loc_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to


    async def scout(self):
        if len(self.units(OBSERVER)) > 0:
            scout = self.units(OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_loc_variance(enemy_location)
                print(move_to)
                await self.do(scout.move(move_to))

        else:
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        draw_dict = {
                     NEXUS: [15, (0,255,0)],
                     ROBOTICSFACILITY: [5, (215, 155, 0)],
                     PYLON: [3, (20,235,0)],
                     ASSIMILATOR: [2, (55, 200, 0)],
                     PROBE: [1, (55, 200, 0)],
                     GATEWAY: [7, (200, 100, 0)],
                     CYBERNETICSCORE: [5, (150, 150, 0)],
                     STARGATE: [8, (255, 0, 0)],
                     VOIDRAY: [3, (255, 100, 0)]
                    }
        for n in draw_dict:
            for nn in self.units(n).ready:
                pos = nn.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[n][0], draw_dict[n][1], -1)

        enemy_headquaters = ["nexus","commandcenter", "hatchery"]
        for e in self.known_enemy_structures:
            pos = e.position
            if e.name.lower() not in enemy_headquaters:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212),-1)
        for e in self.known_enemy_structures:
            pos = e.position
            if e.name.lower() in enemy_headquaters:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for e in self.known_enemy_units:
            if not e.is_structure:
                units = ["probe", "scv", "drone"]

                pos = e.position
                if e.name.lower() in units:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0


        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap-self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0


        cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        self.flipped = cv2.flip(game_data, 0)
        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def build_probe(self):
        if (len(self.units(NEXUS))*15) > len(self.units(PROBE)) and len(self.units(PROBE)) < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.noqueue:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))


    async def build_pylons(self):
        if self.supply_left < 5:
            if self.can_afford(PYLON) and not self.already_pending(PYLON):
                nexus = self.units(NEXUS).ready
                if nexus.exists:
                    await self.build(PYLON, near=nexus.first)


    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vespene = self.state.vespene_geyser.closer_than(25.0, nexus)
            for vesp in vespene:
                if self.can_afford(ASSIMILATOR) and not self.already_pending(ASSIMILATOR):
                    worker = self.select_build_worker(vesp.position)
                    if worker is None or self.units(ASSIMILATOR).closer_than(1.0, vesp).exists:
                        break
                    else:
                        await self.do(worker.build(ASSIMILATOR, vesp))

    async def expand(self):
        if self.units(NEXUS).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(NEXUS):
            await self.expand_now()

    async def build_army_buildings(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if not self.units(GATEWAY).exists:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near = pylon)
            elif self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE).exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near = pylon)
            if not self.units(STARGATE).exists and self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near= pylon)
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.units(ROBOTICSFACILITY).amount < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)
            if self.units(GATEWAY).amount < 1: #((self.iteration / self.ITERATIONS_PER_MINUTE) / 2):
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            if self.units(STARGATE).amount < (self.iteration / self.ITERATIONS_PER_MINUTE):
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near= pylon)

    async def build_army(self):
        for s in self.units(STARGATE).ready.noqueue:
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(s.train(VOIDRAY))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]


    async def attack(self):
        if len(self.units(VOIDRAY).idle) > 0:

            target = False
            if self.iteration > self.do_something_after:
                if self.use_model:
                    prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                    choice = np.argmax(prediction[0])
                    #print('prediction: ',choice)

                    choice_dict = {0: "No Attack!",
                                   1: "Attack close to our nexus!",
                                   2: "Attack Enemy Structure!",
                                   3: "Attack Eneemy Start!"}

                    print("Choice #{}:{}".format(choice, choice_dict[choice]))

                else:
                    choice = random.randrange(0, 4)

                if choice == 0:
                    # no attack
                    wait = random.randrange(20, 165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1:
                    #attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))

                elif choice == 2:
                    #attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    #attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))
                y = np.zeros(4)
                y[choice] = 1
                print(y)
                self.train_data.append([y,self.flipped])

for i in range(100):
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, tiny(use_model= True)), Computer(Race.Terran, Difficulty.Easy)], realtime=False)
