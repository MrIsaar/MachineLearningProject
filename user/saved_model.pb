??8
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
DenseBincount
input"Tidx
size"Tidx
weights"T
output"T"
Tidxtype:
2	"
Ttype:
2	"
binary_outputbool( 
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
$

LogicalAnd
x

y

z
?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.22v2.6.1-9-gc2363d6d0258??4
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:	*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? @*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	? @*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6471*
value_dtype0	
n
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6523*
value_dtype0	
n
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6575*
value_dtype0	
n
hash_table_3HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6627*
value_dtype0	
n
hash_table_4HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6679*
value_dtype0	
n
hash_table_5HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6731*
value_dtype0	
n
hash_table_6HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6783*
value_dtype0	
n
hash_table_7HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6835*
value_dtype0	
n
hash_table_8HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6887*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? @*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	? @*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? @*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	? @*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_8Const*
_output_shapes
: *
dtype0	*
value	B	 R 
|
Const_9Const*
_output_shapes

:	*
dtype0*=
value4B2	"$??H??~?~?B@?LXBl?F??E?
?DL?xD	?YA
}
Const_10Const*
_output_shapes

:	*
dtype0*=
value4B2	"$??QP??;??-Bd?F?V?N???M??-K???J?2C
??
Const_11Const*
_output_shapes	
:?*
dtype0*č
value??B???B100% Orange JuiceB60 Seconds!B7 Days to DieBA Bird StoryBA Hat in TimeBA Story About My UncleBABZUBACE COMBAT™ 7: SKIES UNKNOWNBAPB ReloadedBARK: Survival EvolvedB	ASTRONEERBATLASBAbsolverBAdVenture CapitalistBAge of Empires II HDB)Age of Empires® III: Complete CollectionB"Age of Mythology: Extended EditionBAge of Wonders IIIBAirMech StrikeB	Alan WakeBAlien Swarm: Reactive DropBAlien: IsolationBAliens vs. Predator™BAmerica's Army: Proving GroundsBAmerican Truck SimulatorBAmnesia: A Machine for PigsBAmnesia: The Dark DescentBAnno 2070™BAntichamberBArcheBlade™BArgoBArma 2: Operation ArrowheadBArma 3BArmelloBArt of War: Red TidesBArtifactB!Assassin's Creed 2 Deluxe EditionBAssassin's Creed® OdysseyBAssassin's Creed® OriginsBAssassin's Creed® RevelationsBAssassin's Creed® SyndicateBAssassin's Creed® UnityB+Assassin's Creed™: Director's Cut EditionB Assassin’s Creed® BrotherhoodB%Assassin’s Creed® IV Black Flag™BAssassin’s Creed® RogueBAssetto CorsaBAura KingdomBAwesomenauts - the 2D mobaB
BATTLETECHBBEEPBBLOCKADE 3DBBRAIN / OUTBBad Rats: the Rats' RevengeBBaldur's Gate: Enhanced EditionBBallistic OverkillBBanishedBBastionBBatman - The Telltale SeriesB.Batman: Arkham Asylum Game of the Year EditionB.Batman: Arkham City - Game of the Year EditionBBatman™: Arkham KnightBBatman™: Arkham OriginsBBattle BrothersBBattleBlock Theater®B
BattlebornBBattlefield: Bad Company™ 2B
BattleriteB	BayonettaBBeamNG.driveBBeat HazardB
Beat SaberBBeholderBBendy and the Ink Machine™BBesiegeBBinary DomainBBioShock InfiniteBBioShock™ RemasteredBBit Blaster XLBBlack Desert OnlineB
Black MesaBBlack SquadB	BlackwakeBBlock N LoadB
BlockstormBBlood and BaconBBloons TD BattlesBBook of DemonsBBorderlands 2BBorderlands: The Pre-SequelBBraidB
BrawlhallaBBroforceB
Broken AgeBBrothers - A Tale of Two SonsBBrutal LegendBBully: Scholarship EditionB"Burnout Paradise: The Ultimate BoxB2Business Tour - Board Game with Online MultiplayerBCPUCores :: Maximize Your FPSBCS2DBCall of Duty: World at WarB"Call of Duty® 4: Modern Warfare®B/Call of Duty®: Advanced Warfare - Gold EditionBCall of Duty®: Black OpsBCall of Duty®: Black Ops IIBCall of Duty®: Black Ops IIIBCall of Duty®: GhostsB Call of Duty®: Infinite WarfareB"Call of Duty®: Modern Warfare® 2BCall of Duty®: WWIIBCall to ArmsBCar Mechanic Simulator 2015BCar Mechanic Simulator 2018B
Carpe DiemBCastle Crashers®BCastleMiner ZBCave Story+BCelesteBChild of LightBChivalry: Medieval WarfareBCities: SkylinesBClicker HeroesBClustertruckBCodename CUREBComedy NightBCommand & Conquer: Red Alert 3B"Company of Heroes - Legacy EditionBCompany of Heroes 2BConan ExilesB	ContagionBContrastBCook, Serve, Delicious!B
Cossacks 3BCounter-Strike Nexon: ZombiesBCraft The WorldBCreative DestructionBCreativerseB	CrossCodeBCrossoutBCrusader Kings IIBCrush CrushBCry of FearBCrypt of the NecroDancerBCrysisBCrysis 2 - Maximum EditionBCuisine RoyaleBCupheadBDARK SOULS™ IIB*DARK SOULS™ II: Scholar of the First SinBDARK SOULS™ IIIBDARK SOULS™: REMASTEREDBDC Universe™ OnlineBDCS World Steam EditionB)DEAD OR ALIVE 5 Last Round: Core FightersBDEEP SPACE WAIFUBDISTRAINT: Deluxe EditionB	DLC QuestBDOOMBDRAGON BALL FighterZBDRAGON BALL XENOVERSEBDRAGON BALL XENOVERSE 2BZDYNASTY WARRIORS 8: Xtreme Legends Complete Edition / 真・三國無双７ with 猛将伝BDanganronpa 2: Goodbye DespairB Danganronpa: Trigger Happy HavocBDark Messiah of Might & MagicBDark and LightBDarkest Dungeon®B"Darksiders II Deathinitive EditionBDarksiders Warmastered EditionBDarwin ProjectBDay of InfamyBDayZB	Dead BitsB
Dead CellsBDead Frontier 2B Dead Rising 3 Apocalypse EditionB
Dead SpaceBDead Space™ 2BDead by DaylightB	DeadlightBDeceitBDeep Rock GalacticBDefianceBDefy Gravity ExtendedBDemocracy 3BDeponiaBDepthB	DetentionB!Deus Ex: Game of the Year EditionB*Deus Ex: Human Revolution - Director's CutBDeus Ex: Mankind DividedBDevil DaggersBDevil May Cry 5B!Devil May Cry® 4 Special EditionB
DiRT RallyBDigger OnlineB
Dino D-DayBDirty Bomb®B
DishonoredBDishonored 2B)Divinity: Original Sin - Enhanced EditionB-Divinity: Original Sin 2 - Definitive EditionBDmC: Devil May CryBDoki Doki Literature Club!BDominaBDon't Starve TogetherBDoor KickersBDouble Action: BoogalooBDownwellBLDr. Langeskov, The Tiger, and The Terribly Cursed Emerald: A Whirlwind HeistBDragon Age: OriginsB&Dragon Age: Origins - Ultimate EditionBDragon's Dogma: Dark ArisenBDragoniaB	Duck GameBDuke Nukem ForeverBDungeon DefendersBDungeon Defenders IIBDungeon of the Endless™B
Dungeons 3BDust: An Elysian TailBDying LightBE.Y.E: Divine CybermancyB1EARTH DEFENSE FORCE 4.1 The Shadow of New DespairBELEXB
EVE OnlineBEVERSPACE™BElite DangerousBElswordBEmily is AwayBEmpyrion - Galactic SurvivalBEnclaveBEndless Legend™BEndless Space® - CollectionBEndless Space® 2BEnter the GungeonBEternal SeniaBEuro Truck Simulator 2BEuropa Universalis IVBEverlasting SummerBEvolandBF.E.A.R.B
F.E.A.R. 3BFEZBFINAL FANTASY VIIBFINAL FANTASY VIIIBFINAL FANTASY X/X-2 HD RemasterBFINAL FANTASY XIV OnlineB FINAL FANTASY XV WINDOWS EDITIONBFINAL FANTASY® XIIIBFOR HONOR™BFTL: Faster Than LightBFable - The Lost ChaptersBFable AnniversaryBFactorioBFaeriaB+Fallout 2: A Post Nuclear Role Playing GameB	Fallout 3B#Fallout 3: Game of the Year EditionB	Fallout 4BFallout ShelterB)Fallout: A Post Nuclear Role Playing GameBFallout: New VegasB	Far Cry 3BFar Cry 3 - Blood DragonBFar Cry® 2: Fortune's EditionBFar Cry® 4BFar Cry® 5BFar Cry® PrimalBFarm TogetherBFarming Simulator 15BFarming Simulator 17BFarming Simulator 19BFinding ParadiseB	FirewatchBFishing PlanetBFistful of FragsBFive Nights at Freddy'sBFive Nights at Freddy's 2BFive Nights at Freddy's 3BFive Nights at Freddy's 4B(Five Nights at Freddy's: Sister LocationBFlatOut 2™BFootball Manager 2018BFor The KingBFoxholeBFran BowB#Freddy Fazbear's Pizzeria SimulatorBFreeman: Guerrilla WarfareBFreestyle 2: Street BasketballBFriday the 13th: The GameB	FrostpunkBFuriBGRID 2BGRID AutosportBGRISBGalactic Civilizations IIIBGame Dev TycoonBGang BeastsBGaokao.Love.100DaysBGarfield KartBGarry's ModBGauntlet™ Slayer EditionBGear UpBGenital JoustingBGeometry DashB"Getting Over It with Bennett FoddyBGoat SimulatorBGolf With Your FriendsB	Gone HomeBGorogoaB"Gotham City Impostors Free to PlayBGrand Theft Auto IIIBGrand Theft Auto IVBGrand Theft Auto VB,Grand Theft Auto: Episodes from Liberty CityBGrand Theft Auto: San AndreasBGrand Theft Auto: Vice CityBGraveyard KeeperB	Grim DawnBGrim Fandango RemasteredBGuacamelee! Gold EditionBGunZ 2: The Second DuelBGunpointBGuns of Icarus OnlineB HELLDIVERS™ A New Hell EditionBHITMAN™ 2BHacknetBHammerwatchBHand SimulatorBHand of FateBHatoful BoyfriendBHatredBHearts of Iron IVBHellblade: Senua's SacrificeBHentai GirlB	Her StoryB
Hero SiegeBHeroes & GeneralsB,Heroes® of Might & Magic® III - HD EditionBHidden FolksBHitman: Absolution™BHitman: Blood MoneyBHoldfast: Nations At WarBHollow KnightB	HomefrontBHomefront®: The RevolutionBHomeworld Remastered CollectionBHookBHotline MiamiBHotline Miami 2: Wrong NumberBHouse FlipperBHow to SurviveBHow to Survive 2BHuman: Fall FlatBHuniePopBHunt ShowdownB	HurtworldBHyper Light DrifterB}Hyperdimension Neptunia Re;Birth1 / 超次次元ゲイム ネプテューヌRe;Birth1 / 超次次元遊戲戰機少女重生1B
I am BreadBICEYBINSIDEBInfestation: The New ZB)Injustice: Gods Among Us Ultimate EditionB
InsurgencyBInsurgency: SandstormBInterstellar MarinesBInto the BreachB
Iron SnoutBJazzpunk: Director's CutBJotun: Valhalla EditionBJurassic World EvolutionBJust Cause 2BJust Cause 2: Multiplayer ModBJust Cause™ 3B Keep Talking and Nobody ExplodesBKenshiBKerbal Space ProgramB"Killer is Dead - Nightmare EditionBKilling Floor 2BKingdom Come: DeliveranceBKingdom RushBKingdom: ClassicBKingdom: New LandsBKingdoms and CastlesB Kingdoms of Amalur: Reckoning™B
L.A. NoireBLEGO® Marvel™ Super HeroesBLEGO® WorldsBLIMBOBLISABLYNEB
Late ShiftBLayers of FearBLegend of GrimrockBLethal LeagueBLife is Feudal: Your OwnBLife is Strange - Episode 1BLife is Strange 2B!Life is Strange: Before the StormBLine of SightBLittle InfernoBLittle NightmaresBLoading Screen SimulatorBLong Live The QueenBLords Of The Fallen™BLost CastleBLuciusB"Lucy -The Eternity She Wished For-BMANDAGONBMETAL GEAR RISING: REVENGEANCEB!METAL GEAR SOLID V: GROUND ZEROESB$METAL GEAR SOLID V: THE PHANTOM PAINBMETAL SLUG 3BMONSTER HUNTER: WORLDBMachinariumBMad MaxBMafia IIB	Mafia IIIBMagic DuelsBMagiciteBMagickaB	Magicka 2BMapleStory 2BMass EffectBMass Effect 2BMaterial GirlBMax Payne 3BMechWarrior Online™ Solaris 7BMelody's EscapeBMen of War: Assault Squad 2BMetin2BMetro 2033 ReduxBMetro: Last Light ReduxB+Microsoft Flight Simulator X: Steam EditionB!Middle-earth™: Shadow of War™B/Minecraft: Story Mode - A Telltale Games SeriesB
Mini MetroBMinion MastersBMirrorBMirror's Edge™B
MiscreatedBMitos.is: The GameB%Momodora: Reverie Under The MoonlightBMonaco: What's Yours Is MineBMontaroBMoonbase AlphaBMortal Kombat XBMotorsport ManagerBMount & Blade: WarbandB Mount & Blade: With Fire & SwordBMount Your FriendsBMountainBMove or DieB	MudRunnerBMurder MinersBMurdered: Soul SuspectBMutant Year Zero: Road to EdenBMy Summer CarBMy Time At PortiaB6NARUTO SHIPPUDEN: Ultimate Ninja STORM 3 Full Burst HDB(NARUTO SHIPPUDEN: Ultimate Ninja STORM 4B1NARUTO SHIPPUDEN: Ultimate Ninja STORM RevolutionBNBA 2K17BNBA 2K18BNEKOPARA Vol. 0BNEKOPARA Vol. 1BNEKOPARA Vol. 2BNEKOPARA Vol. 3BNatural Selection 2BNeed For Speed: Hot PursuitBNether: ResurrectedBNever Alone (Kisima Ingitchuna)BNeverwinterBNext Day: SurvivalBNidhoggBNieR:Automata™BNight in the WoodsB0Nioh: Complete Edition / 仁王 Complete EditionBNo Man's SkyBNo More Room in HellB	NorthgardBNuclear ThroneBOLDTVBORION: PreludeBOctodad: Dadliest CatchBOne Finger Death PunchBOne Piece Pirate Warriors 3BOneShotBOrcs Must Die!BOrcs Must Die! 2B,Ori and the Blind Forest: Definitive EditionBOrwell: Keeping an Eye On YouBOsiris: New DawnBOut There SomewhereBOutlastB	Outlast 2B
OvercookedBOvercooked! 2BOxenfreeBOxygen Not IncludedB#PAC-MAN™ Championship Edition DX+BPAYDAY 2BPAYDAY™ The HeistBPC Building SimulatorBPLAYERUNKNOWN'S BATTLEGROUNDSBPOSTAL 2BPRICEBPaint the Town RedB
Paladins®BPapers, PleaseB
Party HardBPath of ExileBPathfinder: KingmakerBPillars of EternityB Pillars of Eternity II: DeadfireBPit People®BPlague Inc: EvolvedBPlanet CoasterBPlanetSide 2BPlanetary Annihilation: TITANSB
PlanetbaseBPlants vs. Zombies GOTY EditionBPoly BridgeBPony IslandBPortal KnightsBPreyBPrimal CarnageBPrison ArchitectBProject CARSBProject CARS 2BProject ZomboidBPrototype 2BPsychonautsB
Punch ClubBPyreBQuake ChampionsBQuake Live™BQuantum BreakBRAGEB RESIDENT EVIL 2 / BIOHAZARD RE:2B5RESIDENT EVIL 7 biohazard / BIOHAZARD 7 resident evilBRIFTBRUINERBRUNNING WITH RIFLESBRWBY: Grimm EclipseB	Rabi-RibiBRace The SunBRaceRoom Racing ExperienceBRadical HeightsBRaftB
RavenfieldBRealm RoyaleBRealm of the Mad GodBRebel GalaxyBRecettear: An Item Shop's TaleBRed Crucible®: FirestormB7Red Orchestra 2: Heroes of Stalingrad with Rising StormBRefunctBReign Of KingsBReignsBRelic Hunters ZeroBRemember MeB%Resident Evil / biohazard HD REMASTERBResident Evil 6 / Biohazard 6B1Resident Evil Revelations / Biohazard RevelationsB5Resident Evil Revelations 2 / Biohazard Revelations 2B!Resident Evil™ 5/ Biohazard 5®BRiders of IcarusBRimWorldBRing of ElysiumB!Rise of Nations: Extended EditionBRise of the Tomb Raider™BRising Storm 2: VietnamBRisk of RainBRisk of Rain 2BRivals of AetherBRoad RedemptionB	RobocraftBRocket League®B%Rocksmith® 2014 Edition - RemasteredBRogue LegacyBRome: Total War™ - CollectionBRustBRyse: Son of RomeB&S.K.I.L.L. - Special Force 2 (Shooter)BS.T.A.L.K.E.R.: Call of PripyatBS.T.A.L.K.E.R.: Clear SkyB#S.T.A.L.K.E.R.: Shadow of ChernobylBSCP: Secret LaboratoryBSCUMBSMITE®BSNOWBSOMABSPORE™B-STAR WARS™ - Knights of the Old Republic™B&STAR WARS™ Empire at War - Gold PackB*STAR WARS™ Jedi Knight - Jedi Academy™BBSTAR WARS™ Knights of the Old Republic™ II - The Sith Lords™B!STAR WARS™ Republic Commando™BSTEINS;GATEBSUNLESS SEABSUPERHOTBSaints Row 2BSaints Row IVBSaints Row: Gat out of HellBSaints Row: The ThirdBSakura ClickerBSakura SpiritBSalt and SanctuaryB	Sanctum 2BScrap MechanicBScribblenauts UnlimitedBSecret World LegendsBSekiro™: Shadows Die TwiceBSerenaBSerious Sam 2BSerious Sam 3: BFEB$Shadow Tactics: Blades of the ShogunBShadow WarriorBShadow Warrior 2BShadow of the Tomb RaiderBShadowrun ReturnsBShadowverse CCGBShakes and FidgetBShellShock LiveBShovel Knight: Treasure TroveBFShower With Your Dad Simulator 2015: Do You Still Shower With Your DadBSid Meier's Civilization® VB+Sid Meier's Civilization®: Beyond Earth™BSid Meier’s Civilization® VIBSimplePlanesB#Sins of a Solar Empire: Rebellion®B
SkullgirlsBSlay the SpireB!Sleeping Dogs: Definitive EditionBSlender: The ArrivalBSlime RancherBSniper Elite 3BSniper Elite 4BSniper: Ghost Warrior 2BSonic Adventure 2BSonic Generations CollectionBSonic ManiaBSoulWorker - Anime Action MMOBSource FilmmakerB)South Park™: The Fractured But Whole™B$South Park™: The Stick of Truth™BSpace EngineersBSpec Ops: The LineBSpeedRunnersBSpintires®: The Original GameBSpiral KnightsBSpooky's Jump Scare MansionBSquadBStar ConflictBStar Trek OnlineB(Star Wars: Battlefront 2 (Classic, 2005)B	StarboundBStardew ValleyBSteamWorld DigBSteep™B	StellarisBStick Fight: The GameBStonehearthBStories: The Path of DestiniesBStranded DeepBStreet Fighter VBStreets of RogueBStronghold Crusader 2BStronghold Crusader HDBStronghold KingdomsBStyx: Master of ShadowsB
SubnauticaBSubnautica: Below ZeroBSuper Crate BoxBSuper HexagonBSuperflightBSupreme Commander 2B"Supreme Commander: Forged AllianceBSurgeon SimulatorB	SurvariumB
Sven Co-opBSword Art Online: Fatal BulletBSystem Shock 2BTEKKEN 7B'THE KING OF FIGHTERS XIII STEAM EDITIONBTabletop SimulatorBTales from the BorderlandsBTales of Berseria™BTales of ZestiriaB	TerraTechBTerrariaB(The Awesome Adventures of Captain SpiritBThe Banner SagaBThe Beginner's GuideBThe Binding of IsaacBThe Binding of Isaac: RebirthBThe Bureau: XCOM DeclassifiedBThe Crew™BThe Crew™ 2BThe Darkness IIB;The Elder Scrolls III: Morrowind® Game of the Year EditionB9The Elder Scrolls IV: Oblivion® Game of the Year EditionBThe Elder Scrolls V: SkyrimB+The Elder Scrolls V: Skyrim Special EditionBThe Elder Scrolls® OnlineBThe Elder Scrolls®: Legends™BThe EscapistsBThe Escapists 2BThe Evil WithinBThe Evil Within 2BThe ExpendabrosB
The ForestB(The Incredible Adventures of Van HelsingBThe IsleBThe Long DarkBThe Lord of the Rings Online™B!The Mean Greens - Plastic WarfareBThe PlanBThe RoomBThe Room TwoBThe Ship: Murder PartyBThe Silent AgeBThe Sims™ 3BThe Stanley ParableBThe SwapperBThe Talos PrincipleBThe Vanishing of Ethan CarterBThe Way of Life Free EditionB2The Witcher 2: Assassins of Kings Enhanced EditionB,The Witcher: Enhanced Edition Director's CutBThe Witcher® 3: Wild HuntBThe WitnessBThe Wolf Among UsBThey Are BillionsBThiefBThis Is the PoliceBThis War of MineBThomas Was AloneB	TimbermanBTitan Quest Anniversary EditionBTo the MoonB$Tom Clancy's Ghost Recon® WildlandsB Tom Clancy's Rainbow Six® SiegeB&Tom Clancy’s Splinter Cell BlacklistBTom Clancy’s The Division™BTomb RaiderBTorchlight IIBToribashBTotal War: ATTILAB(Total War: EMPIRE – Definitive EditionB-Total War: MEDIEVAL II – Definitive EditionB*Total War: NAPOLEON – Definitive EditionBTotal War: WARHAMMERBTotal War: WARHAMMER IIB'Total War™: ROME II - Emperor EditionBTotally Accurate BattlegroundsBTower UniteBTown of SalemBTrackMania Nations ForeverBTrain Simulator 2019BTransformiceB
TransistorBTransmissions: Element 120BTransport FeverBTree of LifeBTree of Savior (English Ver.)BTribes: AscendBTricolour LovestoryBTrine 2: Complete StoryBTrine Enchanted EditionB	Tropico 4B	Tropico 5BTroveBTwo Point HospitalBTyrannyBUltimate Chicken HorseBUltimate Custom NightBUltra Street Fighter® IVBUnEpicB	UndertaleBUnheardBUniverse Sandbox ²BUnturnedB(VA-11 Hall-A: Cyberpunk Bartender ActionBVRChatBVVVVVVBVValiant Hearts: The Great War™ / Soldats Inconnus : Mémoires de la Grande Guerre™BValkyria Chronicles™BValleyB$Vampire: The Masquerade - BloodlinesBVampyrBVerdunBVictor Vran ARPGBVictoria IIBViridiBViscera Cleanup DetailBWAKFUBWARMODEBWallpaper EngineBWar ThunderBWarfaceBWarframeBWargame: Red DragonB Warhammer 40,000: Dawn of War IIB!Warhammer 40,000: Dawn of War IIIB!Warhammer 40,000: Eternal CrusadeBWarhammer 40,000: Space MarineB!Warhammer: End Times - VermintideBWarhammer: Vermintide 2B-Warhammer® 40,000: Dawn of War® - SoulstormBWasteland 2: Director's CutBWatch_Dogs® 2BWatch_Dogs™BWe Were HereBWhat Remains of Edith FinchBWho's Your DaddyB Wolfenstein II: The New ColossusBWolfenstein: The New OrderBWolfenstein: The Old BloodBWorld War 3BWorld of Guns: Gun DisassemblyBWorld of Tanks Blitz MMOBWorld of WarshipsB	WreckfestBXCOM: Enemy UnknownBXCOM® 2BYakuza 0BYet Another Zombie DefenseBYou Have to Win the GameBYoutubers LifeBYu-Gi-Oh! Duel LinksBZ1 Battle RoyaleBZombie Army TrilogyBZombie Panic! SourceBZup!BZup! 2Bresident evil 4 / biohazard 4Bthe static speaks my nameBtheHunter ClassicBtheHunter: Call of the Wild™B!中国式家长 / Chinese ParentsB古剑奇谭三(Gujian3)B 太吾绘卷 The Scroll Of TaiwuB探灵笔记-1v5(Notes of Soul)B0東方天空璋 ～ Hidden Star in Four Seasons.
?5
Const_12Const*
_output_shapes	
:?*
dtype0	*?5
value?5B?5	?"?5                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      
?@
Const_13Const*
_output_shapes	
:?*
dtype0*?@
value?@B?@?B
2006-07-11B
2006-10-11B
2006-10-25B
2006-11-15B
2006-11-29B
2006-12-21B
2007-03-15B
2007-03-20B
2007-03-22B
2007-03-29B
2007-07-17B
2007-08-28B
2007-11-12B
2008-01-04B
2008-03-07B
2008-04-09B
2008-04-16B
2008-09-15B
2008-09-16B
2008-09-17B
2008-10-03B
2008-10-21B
2008-10-22B
2008-10-28B
2008-11-18B
2008-12-02B
2008-12-19B
2009-01-07B
2009-01-08B
2009-01-09B
2009-01-14B
2009-02-18B
2009-03-04B
2009-03-12B
2009-04-10B
2009-05-05B
2009-06-16B
2009-07-02B
2009-07-08B
2009-07-20B
2009-08-19B
2009-09-05B
2009-09-15B
2009-09-16B
2009-10-16B
2009-11-05B
2009-11-11B
2009-12-17B
2010-01-28B
2010-02-11B
2010-02-16B
2010-02-25B
2010-03-01B
2010-03-02B
2010-03-04B
2010-03-23B
2010-03-26B
2010-03-31B
2010-04-12B
2010-04-15B
2010-05-21B
2010-05-25B
2010-06-29B
2010-07-06B
2010-08-30B
2010-09-07B
2010-09-08B
2010-09-10B
2010-09-23B
2010-10-21B
2010-10-29B
2010-11-08B
2010-12-14B
2010-12-15B
2011-01-10B
2011-01-25B
2011-01-27B
2011-03-01B
2011-03-14B
2011-03-17B
2011-03-22B
2011-04-08B
2011-05-03B
2011-05-06B
2011-05-16B
2011-06-09B
2011-06-14B
2011-06-21B
2011-07-29B
2011-08-02B
2011-08-16B
2011-09-01B
2011-09-08B
2011-09-13B
2011-09-28B
2011-10-03B
2011-10-11B
2011-10-18B
2011-10-20B
2011-11-02B
2011-11-03B
2011-11-08B
2011-11-10B
2011-11-17B
2011-11-22B
2011-11-30B
2011-12-06B
2011-12-19B
2012-01-05B
2012-01-31B
2012-02-09B
2012-02-14B
2012-02-16B
2012-02-20B
2012-02-27B
2012-04-11B
2012-04-16B
2012-04-26B
2012-05-31B
2012-06-06B
2012-06-12B
2012-06-27B
2012-06-28B
2012-07-04B
2012-07-10B
2012-07-26B
2012-07-30B
2012-08-01B
2012-08-06B
2012-08-23B
2012-08-29B
2012-08-30B
2012-09-07B
2012-09-14B
2012-09-20B
2012-09-26B
2012-10-11B
2012-10-12B
2012-10-16B
2012-10-23B
2012-10-25B
2012-10-26B
2012-10-29B
2012-10-30B
2012-11-02B
2012-11-12B
2012-11-19B
2012-11-20B
2012-11-27B
2012-11-28B
2013-01-16B
2013-01-24B
2013-01-31B
2013-02-12B
2013-02-15B
2013-02-26B
2013-02-27B
2013-03-04B
2013-03-11B
2013-03-18B
2013-03-21B
2013-03-25B
2013-04-04B
2013-04-09B
2013-04-16B
2013-04-19B
2013-04-24B
2013-04-25B
2013-05-01B
2013-05-10B
2013-05-15B
2013-05-22B
2013-05-23B
2013-05-24B
2013-05-27B
2013-05-30B
2013-06-03B
2013-06-06B
2013-06-25B
2013-06-27B
2013-07-02B
2013-07-04B
2013-07-25B
2013-08-01B
2013-08-02B
2013-08-08B
2013-08-12B
2013-08-13B
2013-08-15B
2013-08-22B
2013-08-23B
2013-08-29B
2013-09-02B
2013-09-03B
2013-09-04B
2013-09-10B
2013-09-12B
2013-09-13B
2013-09-24B
2013-09-26B
2013-10-04B
2013-10-08B
2013-10-11B
2013-10-14B
2013-10-17B
2013-10-21B
2013-10-23B
2013-10-24B
2013-10-25B
2013-10-28B
2013-10-31B
2013-11-08B
2013-11-15B
2013-11-19B
2013-11-29B
2013-12-05B
2013-12-09B
2013-12-13B
2013-12-16B
2014-01-06B
2014-01-09B
2014-01-13B
2014-01-14B
2014-01-22B
2014-01-28B
2014-01-29B
2014-01-30B
2014-02-03B
2014-02-07B
2014-02-14B
2014-02-18B
2014-02-27B
2014-03-03B
2014-03-06B
2014-03-17B
2014-03-25B
2014-03-28B
2014-03-31B
2014-04-01B
2014-04-17B
2014-04-25B
2014-04-29B
2014-05-01B
2014-05-06B
2014-05-08B
2014-05-09B
2014-05-13B
2014-05-15B
2014-05-16B
2014-05-19B
2014-05-20B
2014-05-21B
2014-05-23B
2014-05-26B
2014-05-28B
2014-06-03B
2014-06-04B
2014-06-05B
2014-06-07B
2014-06-09B
2014-06-12B
2014-06-25B
2014-06-26B
2014-06-27B
2014-07-01B
2014-07-02B
2014-07-09B
2014-07-25B
2014-07-28B
2014-07-29B
2014-08-05B
2014-08-07B
2014-08-18B
2014-08-19B
2014-08-27B
2014-08-29B
2014-09-04B
2014-09-05B
2014-09-12B
2014-09-15B
2014-09-17B
2014-09-18B
2014-09-22B
2014-09-23B
2014-09-25B
2014-10-06B
2014-10-07B
2014-10-08B
2014-10-09B
2014-10-10B
2014-10-13B
2014-10-16B
2014-10-20B
2014-10-23B
2014-10-27B
2014-10-28B
2014-10-30B
2014-11-03B
2014-11-04B
2014-11-07B
2014-11-10B
2014-11-11B
2014-11-14B
2014-11-18B
2014-11-19B
2014-11-24B
2014-11-25B
2014-12-01B
2014-12-11B
2014-12-15B
2014-12-18B
2014-12-19B
2014-12-22B
2014-12-29B
2015-01-19B
2015-01-22B
2015-01-23B
2015-01-26B
2015-01-28B
2015-01-29B
2015-01-30B
2015-02-06B
2015-02-13B
2015-02-17B
2015-02-24B
2015-02-25B
2015-02-26B
2015-03-02B
2015-03-06B
2015-03-09B
2015-03-10B
2015-03-11B
2015-03-15B
2015-03-26B
2015-03-30B
2015-04-01B
2015-04-02B
2015-04-09B
2015-04-13B
2015-04-23B
2015-04-27B
2015-04-28B
2015-04-30B
2015-05-04B
2015-05-05B
2015-05-13B
2015-05-14B
2015-05-18B
2015-05-19B
2015-05-21B
2015-05-25B
2015-05-26B
2015-05-29B
2015-05-30B
2015-06-01B
2015-06-02B
2015-06-04B
2015-06-05B
2015-06-18B
2015-06-23B
2015-06-24B
2015-07-06B
2015-07-07B
2015-07-09B
2015-07-23B
2015-07-24B
2015-07-28B
2015-07-29B
2015-08-05B
2015-08-10B
2015-08-12B
2015-08-17B
2015-08-18B
2015-08-19B
2015-08-24B
2015-08-25B
2015-08-27B
2015-09-01B
2015-09-02B
2015-09-08B
2015-09-15B
2015-09-17B
2015-09-18B
2015-09-21B
2015-09-29B
2015-10-01B
2015-10-06B
2015-10-08B
2015-10-13B
2015-10-15B
2015-10-16B
2015-10-19B
2015-10-20B
2015-10-21B
2015-10-23B
2015-10-27B
2015-11-01B
2015-11-05B
2015-11-06B
2015-11-09B
2015-11-17B
2015-11-18B
2015-11-20B
2015-11-30B
2015-12-01B
2015-12-03B
2015-12-04B
2015-12-05B
2015-12-07B
2015-12-08B
2015-12-10B
2015-12-14B
2015-12-15B
2015-12-17B
2015-12-22B
2016-01-04B
2016-01-08B
2016-01-14B
2016-01-15B
2016-01-19B
2016-01-21B
2016-01-22B
2016-01-26B
2016-01-27B
2016-01-28B
2016-01-29B
2016-02-01B
2016-02-02B
2016-02-04B
2016-02-09B
2016-02-15B
2016-02-18B
2016-02-19B
2016-02-24B
2016-02-25B
2016-02-26B
2016-02-28B
2016-02-29B
2016-03-04B
2016-03-07B
2016-03-14B
2016-03-18B
2016-03-28B
2016-03-31B
2016-04-05B
2016-04-08B
2016-04-11B
2016-04-12B
2016-04-18B
2016-04-19B
2016-04-20B
2016-04-21B
2016-04-27B
2016-05-03B
2016-05-09B
2016-05-12B
2016-05-13B
2016-05-17B
2016-05-19B
2016-05-20B
2016-05-24B
2016-05-31B
2016-06-06B
2016-06-14B
2016-06-16B
2016-06-21B
2016-07-05B
2016-07-06B
2016-07-07B
2016-07-12B
2016-07-18B
2016-07-22B
2016-07-25B
2016-08-02B
2016-08-03B
2016-08-09B
2016-08-11B
2016-08-12B
2016-08-23B
2016-08-24B
2016-08-31B
2016-09-08B
2016-09-15B
2016-09-20B
2016-09-27B
2016-09-28B
2016-09-29B
2016-10-04B
2016-10-06B
2016-10-07B
2016-10-13B
2016-10-18B
2016-10-20B
2016-10-24B
2016-10-27B
2016-11-03B
2016-11-08B
2016-11-09B
2016-11-10B
2016-11-11B
2016-11-17B
2016-11-18B
2016-11-22B
2016-11-28B
2016-11-29B
2016-12-02B
2016-12-05B
2016-12-06B
2016-12-08B
2016-12-22B
2017-01-12B
2017-01-23B
2017-01-26B
2017-01-31B
2017-02-01B
2017-02-02B
2017-02-03B
2017-02-13B
2017-02-15B
2017-02-16B
2017-02-21B
2017-02-24B
2017-03-03B
2017-03-06B
2017-03-07B
2017-03-08B
2017-03-10B
2017-03-16B
2017-03-17B
2017-03-23B
2017-03-24B
2017-03-28B
2017-03-29B
2017-04-03B
2017-04-11B
2017-04-18B
2017-04-20B
2017-04-24B
2017-04-27B
2017-05-04B
2017-05-08B
2017-05-18B
2017-05-22B
2017-05-24B
2017-05-25B
2017-05-29B
2017-05-30B
2017-05-31B
2017-06-01B
2017-06-20B
2017-06-22B
2017-07-07B
2017-07-13B
2017-07-15B
2017-07-18B
2017-07-20B
2017-07-25B
2017-07-26B
2017-07-27B
2017-07-28B
2017-07-31B
2017-08-01B
2017-08-07B
2017-08-17B
2017-08-21B
2017-08-22B
2017-08-24B
2017-08-27B
2017-08-28B
2017-08-29B
2017-08-31B
2017-09-14B
2017-09-20B
2017-09-21B
2017-09-26B
2017-09-28B
2017-09-29B
2017-10-04B
2017-10-05B
2017-10-06B
2017-10-09B
2017-10-12B
2017-10-13B
2017-10-16B
2017-10-17B
2017-10-26B
2017-10-30B
2017-10-31B
2017-11-02B
2017-11-07B
2017-11-08B
2017-11-09B
2017-11-15B
2017-11-16B
2017-11-17B
2017-12-04B
2017-12-06B
2017-12-12B
2017-12-14B
2017-12-21B
2017-12-29B
2018-01-18B
2018-01-23B
2018-01-25B
2018-01-26B
2018-02-01B
2018-02-08B
2018-02-13B
2018-02-22B
2018-02-23B
2018-02-26B
2018-02-27B
2018-02-28B
2018-03-02B
2018-03-06B
2018-03-07B
2018-03-08B
2018-03-09B
2018-03-26B
2018-03-30B
2018-04-10B
2018-04-19B
2018-04-24B
2018-04-27B
2018-04-30B
2018-05-01B
2018-05-08B
2018-05-17B
2018-05-21B
2018-05-23B
2018-06-04B
2018-06-05B
2018-06-11B
2018-06-14B
2018-06-15B
2018-06-25B
2018-06-27B
2018-06-28B
2018-07-25B
2018-08-01B
2018-08-02B
2018-08-06B
2018-08-07B
2018-08-09B
2018-08-10B
2018-08-15B
2018-08-17B
2018-08-29B
2018-09-05B
2018-09-14B
2018-09-19B
2018-09-20B
2018-09-25B
2018-09-26B
2018-09-28B
2018-10-04B
2018-10-05B
2018-10-11B
2018-10-17B
2018-10-19B
2018-11-13B
2018-11-16B
2018-11-19B
2018-11-28B
2018-12-04B
2018-12-06B
2018-12-12B
2018-12-13B
2018-12-14B
2018-12-18B
2018-12-22B
2019-01-15B
2019-01-17B
2019-01-23B
2019-01-24B
2019-01-29B
2019-01-30B
2019-01-31B
2019-02-05B
2019-02-14B
2019-02-28B
2019-03-07B
2019-03-21B
2019-03-28
?+
Const_14Const*
_output_shapes	
:?*
dtype0	*?*
value?*B?*	?"?*                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?n
Const_15Const*
_output_shapes	
:?*
dtype0*?m
value?mB?m?B11 bit studiosB72K Australia;Gearbox Software;Aspyr (Mac);Aspyr (Linux)B=2K Boston;2K Australia;Blind Squirrel;Feral Interactive (Mac)B 2K Czech;Feral Interactive (Mac)B2K MarinB4A GamesB5th Cell MediaB800 North and Digital RanchB@unepic_franBAMPLITUDE StudiosB
Abrakam SAB Adriaan de Jongh;Sylvain TegroegBAfterthought LLCBAirtight GamesBAlexander BruceBAlmost Human GamesBAmanita DesignBAmistech GamesBAnkama StudioB%Antimatter Games;Tripwire InteractiveBAnvil Game StudiosBAquiris Game StudioBArc System WorksBArkane StudiosBArrowhead Game StudiosBArtefacts StudiosBAtelier 801BAurora StudioB	AutomatonBAvalanche StudiosBCAvalanche Studios;Feral Interactive (Mac);Feral Interactive (Linux)BAwesome Games StudioBAxolot GamesBBANDAI NAMCO Studio Inc.BBANDAI NAMCO StudiosBBANDAI NAMCO Studios Inc.BBankroll StudiosBBeam Team GamesBBeamNGBBeamdogB
Beat GamesBBehaviour Digital Inc.BBehaviour Interactive Inc.BBennett FoddyBBerserk GamesB	BetaDwarfBBethesda Game StudiosBBig CorporationBBig Fat AlienBBig Huge Games;38 StudiosBBioWareBBioWare;Aspyr (Mac)BBitbox Ltd.BBithell GamesBBlack Isle StudiosBBlackSpot EntertainmentBBlacklight InteractiveBBlankMediaGamesBBlind Sky StudiosBBloober Team SABBlue Byte;Related DesignsBBlue Isle StudiosBBlue Mammoth GamesBBlue StudioBBohemia InteractiveBBombserviceBBoneloafBBoss Key ProductionsBBossa StudiosBBrace Yourself GamesBBugbearBBugbear EntertainmentBBuried SignalBCAPCOM Co., Ltd.BCCPBCD PROJEKT REDBCI Games;Deck 13BCREATIVE ASSEMBLYBCCREATIVE ASSEMBLY;Feral Interactive (Linux);Feral Interactive (Mac)B)CREATIVE ASSEMBLY;Feral Interactive (Mac)BCCREATIVE ASSEMBLY;Feral Interactive (Mac);Feral Interactive (Linux)BCakeEaterGamesBCampo SantoBCapcomBCapcom Game Studio VancouverBCarbon GamesBCellar Door GamesB/Christian Whitehead;Headcannon;PagodaWest GamesBChucklefishBCity InteractiveBClapfootB&Claudiu Kiss;The Irregular CorporationBClever Endeavour GamesBCodeBrush GamesBKCodemasters Racing Studio;Feral Interactive (Linux);Feral Interactive (Mac)B*Codemasters Racing;Feral Interactive (Mac)BDCodemasters Racing;Feral Interactive (Mac);Feral Interactive (Linux)B
Code}{atchBCoffee Stain StudiosBCold Beam GamesBColossal Order Ltd.BCompulsion GamesBConcernedApeBConchShip GamesB
CrackshellBCrate EntertainmentBCreSpirit;GemaYueB
CreabilityBCreaky Corpse LtdBCCreative Assembly;Feral Interactive (Mac);Feral Interactive (Linux)BCreobitBCriterion GamesBCroteamBCrowbar CollectiveBCrows Crows CrowsBCryptic StudiosBRCrystal Dynamics;Eidos-Montréal;Feral Interactive (Mac);Feral Interactive (Linux)BbCrystal Dynamics;Eidos-Montréal;Feral Interactive (Mac);Feral Interactive (Linux);Nixxes softwareBCrytekBCrytek StudiosB	CtrlMovieBCyanide StudioBCyberCoconut;Fabio FerraraBCyberConnect2 Co. Ltd.BCyberConnect2 Co., Ltd.BCygames, Inc.BDICEBDIMPSBDONTNOD EntertainmentBGDONTNOD Entertainment;Feral Interactive (Mac);Feral Interactive (Linux)BDaedalic EntertainmentBDambuster StudiosBDan FornaceBDaniel Mullins GamesBDarkflow SoftwareBDavid OReillyBDay 1 StudiosBDaybreak Game CompanyB;Deck Nine;Feral Interactive (Mac);Feral Interactive (Linux)BDeep Silver VolitionB*Deep Silver Volition;High Voltage SoftwareBDefiant DevelopmentBDekovir EntertainmentBDennaton GamesB	DesertkunBDestructive CreationsBDevil's DetailsB
DieselmineBDiggerWorld Ltd.BDigital ConfectionersBDigital ExtremesBDigital MelodyBDigitalDNA Games LLCBDigitalmindsoftB	DingalingBDinosaur Polo ClubBDoctor Entertainment ABB
Dodge RollBDolphinBarnBDominique GrieshoferBDotEmuBDouble Action FactoryBDouble Damage GamesBDouble Fine ProductionsBDoubleDutch GamesBDovetail GamesBDragonfly GF Co., LTDBDrinkBox StudiosB
Dry CactusBEA Los AngelesBEA Redwood ShoresBEQ-Games , Pixel Dash StudiosBEagle Dynamics SABEasyGameStationB!Edmund McMillen and Florian HimslB@Eidos Montreal;Feral Interactive (Linux);Feral Interactive (Mac)B&Eidos Montreal;Feral Interactive (Mac)B0Eidos-Montréal;Crystal Dynamics;Nixxes softwareB'Eidos-Montréal;Feral Interactive (Mac)BEko SoftwareBEleon Game StudiosBElias Viglione;Jussi KukkonenBEmpyreanBEndnight Games LtdBEnsemble StudiosBEntrada Interactive LLCBEugen SystemsBEverything Unlimited Ltd.BEvil Mojo GamesBEvil Tortilla GamesBExpansive WorldsB"Expansive Worlds;Avalanche StudiosBFacepunch StudiosBFailbetter GamesBFantaBlade NetworkBFatsharkBFenix Fire EntertainmentB'Firaxis Games;Aspyr (Mac);Aspyr (Linux)B?Firaxis Games;Feral Interactive (Mac);Feral Interactive (Linux)BFireFly StudiosBFireproof GamesBFishing Planet LLCBFistful of Frags TeamBFlippfly LLCBFlying Wild HogBFreakinware StudiosBFredaikis ABB
Free LivesBFreebird GamesBFreejamBFrictional GamesBFromSoftwareBFromSoftware, IncBFromSoftware, Inc.BFrontier DevelopmentsB
FrozenbyteB
FullbrightBFuncomBGSC Game WorldBGaijin EntertainmentBGalactic CafeBGame ScienceBGamepires;CroteamBGas Powered GamesBGearbox SoftwareBGearbox Software;Aspyr (Mac)B*Gearbox Software;Aspyr (Mac);Aspyr (Linux)BGears for BreakfastBGhost Ship GamesBGhost Town Games Ltd.B(Ghost Town Games Ltd.;Team17 Digital LtdB
GhostSharkB
Giant ArmyBGiant SparrowBGiant SquidBGiants SoftwareBGirlGameBGoing Loud StudiosBGone North GamesBGrapeshot Games;Instinct GamesBGreenheart GamesBGrey Havens, LLCBGrinding Gear GamesBGrizzlyGamesB$Gunfire Games;Vigil Games;THQ NordicB	HFM GamesB
HL-GalgameBHaemimont GamesBHanako GamesBHangar 13;Aspyr (Mac)BHarebrained SchemesBHeart MachineBHello GamesBHeroic Leap GamesBHi-Rez StudiosBHinterland Studio Inc.BHoly PriestBHoobalugalar_XBHopoo GamesBHouse On FireBHumble Hearts LLCBHuniePotBHunter StudioBHyper Hippo GamesBCIDEA FACTORY Co., Ltd.;COMPILE HEART Co., Ltd.;FELISTELLA Co., Ltd.BIMCGAMES Co.,Ltd.BIO Interactive A/SBIcetesy SPRLBIllFonicBImage & Form GamesBInfinite FallBInfinity WardBInfinity Ward;Aspyr (Mac)BInterplay Inc.BIntroversion SoftwareBInvent4 EntertainmentBIo-Interactive A/SB*Io-Interactive A/S;Feral Interactive (Mac)B	Ion StormB"Iron Lore Entertainment;THQ NordicBIronOak GamesB%Ironclad Games;Stardock EntertainmentBIronhide Game StudioB8Irrational Games;Aspyr (Mac);Virtual Programming (Linux)B&Irrational Games;Looking Glass StudiosBIvory TowerB5Ivory Tower in collaboration with Ubisoft ReflectionsBJCKSLAPBJForce GamesBJesse MakkonenBJoey Drew Studios Inc.BJoycityBJundroo, LLCB(KADOKAWA GAMES / GRASSHOPPER MANUFACTUREBKAGAMI WORKsBKAIKO;Vigil GamesBKK Game StudioBKOEI TECMO GAMES CO., LTD.BKOGBKaos Studios;Digital ExtremesB
Keen GamesBKeen Software HouseBKillHouse GamesBKillmonday Games ABBKlei EntertainmentBKojima ProductionsBKonami Digital EntertainmentBKrillbite StudioBKristjan SkuttaBKunos SimulazioniBKyle SeeleyBLab Zero GamesBLag StudiosBLandfallBLandfall WestBLandon PodbielskiBLarian StudiosBLazy Bear GamesBLeague of GeeksBLighthouse Games StudioBLion Games Co., Ltd.BLion Shield, LLCBLionhead StudiosBLittle Cat FeetBLittle OrbitBLo-Fi GamesB
Lucas PopeB	LucasArtsBLudeon StudiosBLukewarm MediaBM2H;Blackmill GamesBMAGES. Inc.BMachine GamesBMachineGamesBMaciej Targoni;Wojciech WasiakBMadruga WorksBMasangsoft, Inc.BMassive EntertainmentBMastfire Studios Pty LtdBMatt DabrowskiBMatt Makes Games Inc.BMaxis™B-Mediatonic;Hato Moa;The Irregular CorporationBMega Crit GamesBMesshofBMicroblast GamesBMicrosoft Game StudiosBMilkstone StudiosBMimimi ProductionsBMine Loader Software Co., Ltd.BMiniBossBMinor Key GamesB(Modern Visual Arts Laboratory;SEO INSEOKBMonochrome, IncBMonolith ProductionsBMonolith Productions, Inc.B#Monolith Productions, Inc.;TimegateBMonomi ParkBMoon Studios GmbHBMoonlit WorksBMoppinBMotion TwinBMouldy Toof StudiosB
Muse GamesBMy.comB
NEKO WORKsBNEXT StudiosB	NS STUDIOBNabi StudiosBNadeoBNantGBNdemic CreationsBNecrophone GamesBNeko Climax StudiosBNeocoreGamesBNerialB)NetherRealm Studios;High Voltage SoftwareBNetherRealm Studios;QLOCBNew World InteractiveBNexon Korea CorporationBNicalis, Inc.BNicalis, Inc.;Studio PixelBNickervision StudiosBNight School StudioB
Ninja KiwiBNinja TheoryBNo Brakes GamesBNo More Room in Hell TeamBNoble Empire Corp.BNoioBNoio;LicoriceBNomada StudioBNorthwood StudiosBNovalinkBNumantian GamesBNumber NoneBOVERKILL - a Starbreeze Studio.BOVERKILL SoftwareBObsidian EntertainmentB;Obsidian Entertainment;Aspyr (Mac, Linux, & Windows Update)BOffworld IndustriesB8Olli Harjola, Otto Hantula, Tom Jubert, Carlo CastellanoBOovee® Game StudiosBOrange_JuiceBOsmotic StudiosBOsumia GamesBOuterlight Ltd.BOverhype StudiosBOwlcat GamesBPUBG CorporationBPandemic StudiosBParadox Development StudioBPathea GamesB
Paul FischBPayload StudiosBPearl AbyssB
PetroglyphBPhosphor Games StudioBPieces InteractiveBPinokl Games;KvertaBPiranha BytesBPiranha Games Inc.BPixelTail GamesBPlanetary Annihilation IncBPlatinumGamesBPlaya Games GmbHBPlaydeadBPlayful Corp.B
PlaysaurusBPlaysport GamesBPocketwatch GamesBPolytron CorporationBPopCap Games, Inc.B$Poppermost Productions;WastedStudiosBPositech GamesBPsyonix, Inc.BQLOCB
QLOC;DIMPSBQuiet RiverB	RETO MOTOBROCKFISH GamesBRadiant EntertainmentBRadical EntertainmentBRadical Fish GamesBRaven Software;Aspyr (Mac)BRe-LogicBReactive Drop TeamBRealmforge StudiosB	RebellionBRed BarrelsBRed Dot GamesBRed Hook StudiosBRedCandleGamesBRedbeet InteractiveBReikon GamesBRelicBRelic EntertainmentBERelic Entertainment;Feral Interactive (Mac);Feral Interactive (Linux)B1Relic Entertainment;Feral Interactive (Mac/Linux)BRemedy EntertainmentBRobTop GamesBRobot EntertainmentBRobot GentlemanBRocketeer Games Studio, LLCBRockstar GamesBRockstar New EnglandBRockstar NorthBRockstar North / TorontoBRockstar North;Rockstar TorontoBRockstar StudiosBRocksteady StudiosB*Rocksteady Studios;Feral Interactive (Mac)BRogue SnailBRonimo GamesBRooster Teeth GamesB	RuneStormBRunic GamesBRunning With ScissorsBSANDLOTBSCS SoftwareBSEGABSNK CORPORATIONBSNK CORPORATION;DotEmuB
SOFF GamesBSUPERHOT TeamBSaber InteractiveBSad Panda StudiosB
Sam BarlowBScavengers StudioBScott CawthonBSector3 StudiosBSenscapeBShining Rock Software LLCBShiro GamesBShiver GamesB(Shokunin;Thomas M. Visser;Vincent ThieleBSilver Dollar GamesBSka StudiosBSkyBox Labs;Big Huge GamesBSkyBox Labs;Ensemble StudiosBHSkybox Labs;Hidden Path Entertainment;Ensemble Studios;Forgotten EmpiresB!Sledgehammer Games;Raven SoftwareBSlightly Mad StudiosBSloclapBSmartly Dressed GamesB
SmashGamesBSnail Games USABSnoutUpBSorathBSouth East GamesBSoviet GamesBSparkypants Studios, LLCBSpearhead GamesBSpiderling StudiosB*Spike Chunsoft Co., Ltd.;Abstraction GamesBSpiral Game StudiosBSplash DamageBSports InteractiveBSquadBSquare EnixBSquare Enix;PlatinumGames Inc.BStainless Games Ltd.BStanding Stone Games, LLCBStar Gem Inc.B
StarbreezeBStarbreeze Studios ABBStardock EntertainmentBStartupTim, LLC;Tim SullivanBSteel Crate GamesBSteelRaven7BStegersaurus Software Inc.BStoicBStreum On StudioBStudio MDHR Entertainment Inc.BBStudio Wildcard;Instinct Games;Efecto Studios;Virtual Basement LLCBStunlock StudiosBSubset GamesBSukeban GamesBSupergiant GamesBSuspicious DevelopmentsBSven Co-op TeamBSystem Era SoftworksBTT Games;Traveller's TalesBTaleWorlds EntertainmentBTango GameworksBTargem GamesBTarsier StudiosBTeam Bondi;Rockstar LeedsBTeam CherryBTeam Fractal AlligatorB%Team NINJA;KOEI TECMO GAMES CO., LTD.BTeam PsykskallarBTeam ReptileBTeam SalvatoB&Team17 Digital Ltd;Mouldy Toof StudiosBTechlandBTelltale GamesBTequila Works, S.L.BTerry CavanaghBThe AstronautsBThe Bearded LadiesBThe BehemothBThe Chinese RoomBThe Farm 51BThe Fun PimpsBThe Game BakersBThe Indie StoneBThe Sims StudioBThekla, Inc.BThing TrunkBThomas BowkerBThose Awesome GuysBThunder Lotus GamesBTitan Forge GamesBTitan StudioB"Toadman Interactive;Jagex;ArtplantBTomorrow CorporationBTorn Banner StudiosBTotal Mayhem GamesBTraveller's TalesBTrendy EntertainmentBTreyarchBTreyarch;Aspyr (Mac)BTrion WorldsBTripwire InteractiveBTriumph StudiosBTroika GamesBTwo Point StudiosBU-Play OnlineB	U.S. ArmyBUbisoftBUbisoft - San FranciscoB/Ubisoft Annecy;Ubisoft Montpellier;Ubisoft KievBUbisoft MontpellierBUbisoft MontrealB=Ubisoft Montreal, Massive Entertainment, and Ubisoft ShanghaiB4Ubisoft Montreal, Red Storm, Shanghai, Toronto, KievBHUbisoft Montreal;Red Storm;Ubisoft Shanghai;Ubisoft Toronto;Ubisoft KievB9Ubisoft Montreal;Ubisoft Quebec;Ubisoft Toronto;Blue ByteBUbisoft MontréalBmUbisoft Paris;Ubisoft Annecy;Ubisoft Bucharest;Ubisoft Montpellier;Ubisoft Milan;Reflections;Ubisoft BelgradeB?Ubisoft Quebec, in collaboration with Ubisoft Annecy, Bucharest, Kiev, Montreal, Montpellier, Shanghai, Singapore, Sofia, Toronto studiosBuUbisoft Quebec;Ubisoft Montreal;Ubisoft Bucharest;Ubisoft Singapore;Ubisoft Montpellier;Ubisoft Kiev;Ubisoft ShanghaiBUbisoft San FranciscoBUbisoft Sofia;Ubisoft KievBUbisoft TorontoB*United Front Games;Feral Interactive (Mac)BUnknown Worlds EntertainmentBUnreal SoftwareBUpper One Games;E-Line MediaBUrban GamesBVRChat Inc.BValveB*Valve Corporation, Nexon Korea CorporationBVertigo Gaming Inc.B*Virtual Basement LLC;Code Headquarters LLCBVirtual Heroes;Army Game StudioBVisceral GamesBVisual ConceptsBVlambeerBVolitionBVostok GamesBWB Games Montreal;Splash DamageBWEBZEN Inc.BWargaming Group LimitedBWarhorse StudiosBWarm Lamp GamesBWeMadeBWeappy StudioBWild Shadow Studios;Deca GamesBWinged CloudBWube Software LTD.BX-LegendBYAGERB	YETU GAMEBYacht Club GamesBYoung HorsesBZenimax Online StudiosBZero Point SoftwareB/Zoe Vartanian;Badru;Isa Hutchinson;Michael BellBZombie Panic TeamBid SoftwareBinXile EntertainmentBkChamp GamesBmarbenxBoddonegamesBthe whale husbandBtobyfoxB上海アリス幻樂団B$上海烛龙信息科技有限公司B墨鱼玩游戏B搞快点工作室B高考恋爱委员会
?)
Const_16Const*
_output_shapes	
:?*
dtype0	*?)
value?(B?(	?"?(                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?K
Const_17Const*
_output_shapes	
:?*
dtype0*?J
value?JB?J?B(none)B11 bit studiosB1C EntertainmentB2KB2K;Aspyr (Mac)B2K;Aspyr (Mac);Aspyr (Linux)B2K;Feral Interactive (Mac)B42K;Feral Interactive (Mac);Feral Interactive (Linux)B2K;Missing Link GamesB38 Studios;Electronic ArtsB3909B	505 GamesB800 North and Digital RanchB8FloorB@unepic_franBAGM PLAYISMB
ActivisionBNActivision (Excluding Japan and Asia);FromSoftware (Japan);方块游戏 (Asia)BActivision;Aspyr (Mac)BAdriaan de JonghBAdult Swim GamesBAeria GamesBAfterthought LLCBAlawar PremiumBAlmost Human GamesBAmanita DesignBAmistech GamesBAnkama GamesBAnnapurna InteractiveBAnother IndieBAnvil Game StudiosBAquiris Game StudioBAspyrBAtelier 801B	AutomatonBAvalanche StudiosBAwesome Games StudioBAxolot GamesBBANDAI NAMCO EntertainmentB,BANDAI NAMCO Entertainment;FromSoftware, IncBBankroll StudiosBBeam Team Pty LtdBBeamNGBBeamdogB
Beat GamesBBehaviour Digital Inc.BBehaviour Interactive Inc.BBennett FoddyBBerserk GamesB	BetaDwarfBBethesda SoftworksBBethesda-SoftBBig Fat AlienBBitbox Ltd.BBithell GamesBBlack Maple GamesBBlackSpot EntertainmentBBlankMediaGamesBBlazing Griffin Ltd.BBlind Sky StudiosBBlue Isle StudiosBBohemia InteractiveBBoss Key ProductionsBBossa StudiosB'Brace Yourself Games;Klei EntertainmentBCAPCOM Co., Ltd.BCAPCOM Co., Ltd. BCCPBCD PROJEKT REDBCD PROJEKT RED;1C-SoftClubBCI GamesBCakeEaterGamesBCapcomBCarbon GamesBCarpe Fulgur LLCBCellar Door GamesBChucklefishBClapfootBClever Endeavour GamesBCoconut Island GamesBCodebrush GamesB=Codemasters;Feral Interactive (Linux);Feral Interactive (Mac)B#Codemasters;Feral Interactive (Mac)B=Codemasters;Feral Interactive (Mac);Feral Interactive (Linux)B
Code}{atchBCoffee Stain PublishingBCold Beam GamesBConcernedApeBConchShip GamesB
CrackshellBCrate EntertainmentB
CreabilityBCreaky Corpse LtdBCrowbar CollectiveBCrows Crows CrowsBCrytekBCrytek BCurve DigitalBCyberCoconutB Cygames, Inc.;Cygames Korea Inc.BD3 PUBLISHERBDANKIEBDaedalic EntertainmentBDan FornaceBDaniel Mullins GamesBDaybreak Game CompanyB
Deca GamesB@Deck13;WhisperGames;DANGEN Entertainment;Mayflower EntertainmentBDeep SilverBDefiant DevelopmentBDegicaBDemruthB	DesertkunBDestructive CreationsBDevolver DigitalBDevolver Digital;CroteamBDiggerWorld Ltd.BDigital ConfectionersBDigital ExtremesBDigitalDNA Games LLCBDigitalmindsoftBDingaling Productions, LLCBDinosaur Polo ClubBDoctor Entertainment ABBDolphinBarnBDominique GrieshoferBDouble Action FactoryBDouble Damage GamesBDouble Fine PresentsB"Double Fine Presents;David OReillyBDouble Fine ProductionsBDovetail Games - FlightBDovetail Games - TrainsBDrinkBox StudiosB
Dry CactusBE-Line MediaBEQGamesBEdmund McMillenBElectronic ArtsBEleon Game StudiosBEndnight Games LtdBEntrada Interactive LLCBEuroVideo MedienBEverything Unlimited Ltd.BEvil Tortilla GamesB"Expansive Worlds;Avalanche StudiosBFacepalm GamesBFacepunch StudiosBFailbetter GamesBFatsharkBFellow TravellerBFinjiBFireFly StudiosBFireproof GamesBFish Factory GamesBFishing Planet LLCBFistful of Frags TeamBFlippfly LLCBFocus Home InteractiveBForever Entertainment S. A.BFreakinware StudiosBFredaikis ABBFreebird GamesBFreejamBFrictional GamesB-FromSoftware, Inc.;BANDAI NAMCO EntertainmentB,FromSoftware, Inc;BANDAI NAMCO EntertainmentBFrontier DevelopmentsBFrozen District;PlayWay S.A.B
FrozenbyteBFruitbat FactoryB
FullbrightBFuncomBGSC Game WorldBGSC World PublishingBGaijin Distribution KFTBGaijin EntertainmentBGalactic CafeBGame ScienceBGameforge 4D GmbHBGearbox PublishingBGearbox Publishing;Aspyr (Mac)BGears for BreakfastB
Giant ArmyBGiants SoftwareBGirlGameBGoing Loud StudiosBGood Shepherd EntertainmentBGrapeshot GamesBGreenheart GamesBGrey Havens, LLCBGrinding Gear GamesBGrizzlyGamesBGrunge Games LTDB	Gun MediaB	HFM GamesBHanako GamesBHeart MachineBHello GamesBHi-Rez StudiosBHinterland Studio Inc.BHoly PriestBHoobalugalar_XBHuniePotBHyper Hippo GamesBIMCGAMES Co.,Ltd.BIce Water GamesBIcetesy SPRLB7Idea Factory International, Inc.;IDEA FACTORY Co., Ltd.BImage & Form GamesB	IndieGalaBIo-Interactive A/SB*Io-Interactive A/S;Feral Interactive (Mac)BIronhide Game StudioBJForce GamesBJesse MakkonenBJoey Drew Studios Inc.BJoycityBJundroo, LLCBKK Game StudioBKOEI TECMO GAMES CO., LTD.B	KOG GamesBKakao Games Europe B.V.BKalypso Media DigitalBKeen Software HouseBKillHouse GamesBKillmonday Games ABBKlei EntertainmentBKonami Digital EntertainmentBKrillbite StudioBKristjan SkuttaBKunos SimulazioniBKyle SeeleyBLag StudiosBLandfallBLarian StudiosB
Last LevelBLeague of GeeksBLever GamesBLighthouse Games StudioBLion Shield, LLCBLittle OrbitBLo-Fi GamesB2LucasArts;Aspyr (Mac);Disney Interactive;LucasfilmB2LucasArts;Aspyr (Mac);Lucasfilm;Disney InteractiveB9LucasArts;Disney Interactive;Lucasfilm;Aspyr (Mac, Linux)B&LucasArts;Lucasfilm;Disney InteractiveB&Lucasfilm;LucasArts;Disney InteractiveBLudeon StudiosBM2HBMBDLBMaciej TargoniBMadruga WorksBMarvelous;Autumn GamesBMasangsoft, Inc.BMastfire Studios Pty LtdBMatt Makes Games Inc.BMediascape Co., Ltd.BMega Crit GamesB	Meridian4BMesshofBMicroblast GamesBMicroidsBMicrosoft StudiosBMilkstone StudiosBMinor Key GamesBModern Visual Arts LaboratoryBMonochrome, IncBMonomi ParkBMoonlit WorksBMotion TwinB
Muse GamesBMy.comBNASABNEXT Studios;bilibiliB	NS STUDIOBNVLMakerBNdemic CreationsBNecrophone GamesBNeko Climax StudiosBNeocoreGamesBNether Productions, LLCBNew World InteractiveBNexon AmericaBNexon America Inc.BNexon Korea CorporationBNicalis, Inc.BNickervision StudiosBNight School StudioBNightdive StudiosB
Ninja KiwiBNinja TheoryBNoble Empire Corp.BNorthwood StudiosBNovalinkBNumantian GamesBNumber NoneBOffworld IndustriesBOovee® Game StudiosBOsumia GamesBOverhype StudiosBPUBG CorporationBPanic Art StudiosBPanic;Campo SantoBParadise ProjectBParadox InteractiveBPayload StudiosBPerfect World EntertainmentBPiranha Games Inc.BPixelTail GamesBPlanetary Annihilation IncBPlayStation Mobile, Inc.BPlayWay S.A.BPlaya Games GmbHBPlaydeadBPlayful Corp.B
PlaysaurusBPocketwatch GamesBPopCap Games, Inc.BPositech GamesBPrivate Division BPsyonix, Inc.BQuiet RiverB	RETO MOTOBROCKFISH GamesBRaw FuryBRe-LogicBReactive Drop TeamB	RebellionBRed BarrelsBRed Hook StudiosBRedCandleGames;AGM PLAYISMBRemedy EntertainmentBReverb Triple XPBReverb Triple XP;Circle 5BRobTop GamesBRobot EntertainmentBRobot GentlemanBRocketeer Games Studio, LLCBRockstar GamesBRogue SnailBRonimo GamesBRooster Teeth GamesB	RuneStormBRunic GamesBRunning With ScissorsBSCS SoftwareBSEGAB6SEGA;Feral Interactive (Linux);Feral Interactive (Mac)BSEGA;Feral Interactive (Mac)B6SEGA;Feral Interactive (Mac);Feral Interactive (Linux)B"SEGA;Feral Interactive (Mac/Linux)BSNK CORPORATIONB#SQUARE ENIX;Feral Interactive (Mac)BSUPERHOT TeamBSad Panda StudiosB
SakuraGameB
Sam BarlowBScavengers StudioBScott CawthonB)Sector3 Studios;RaceRoom Entertainment AGBSekai ProjectBSelf PublishedBSenscapeBShining Rock Software LLCBShiro GamesBShiver GamesBSilver Dollar GamesBSka StudiosB/Slightly Mad Studios;BANDAI NAMCO EntertainmentBSmartly Dressed GamesB
SmashGamesBSnail Games USABSnoutUpBSorathBSouth East GamesBSoviet GamesBSpearhead GamesBSpiderling StudiosBSpike Chunsoft Co., Ltd.BSquare EnixB=Square Enix;Feral Interactive (Linux);Feral Interactive (Mac)B#Square Enix;Feral Interactive (Mac)B=Square Enix;Feral Interactive (Mac);Feral Interactive (Linux)B=Square Enix;Feral interactive (Mac);Feral Interactive (Linux)BStanding Stone Games, LLCBStarbreeze Publishing ABBStardock EntertainmentBStartupTim, LLC;Tim SullivanBSteel Crate GamesBSteelRaven7BStegersaurus Software Inc.BStrategy FirstBStreum On StudioBStudio MDHR Entertainment Inc.BStudio WildcardBStunlock StudiosBSubset GamesBSupergiant GamesBSuspicious DevelopmentsBSven Co-op TeamBSystem Era SoftworksBTCH Scarlet LimitedB
THQ NordicBTaleWorlds EntertainmentBTeam CherryBTeam PsykskallarBTeam ReptileBTeam SalvatoBTeam17 Digital LtdBTechland PublishingBTelltale GamesBTerry CavanaghBThe AstronautsBThe BehemothBThe Farm 51BThe Fighter CollectionBThe Fun Pimps Entertainment LLCBThe Game BakersBThe Indie StoneBThe Irregular CorporationBThekla, Inc.BThing TrunkBThomas BowkerBThose Awesome GuysBThunder Lotus GamesBTitan StudioBToadman Interactive;JagexBTomorrow CorporationBTopware InteractiveBTorn Banner StudiosBTotal Mayhem GamesBTrapdoorBTrendy EntertainmentBTrion WorldsBTripwire InteractiveBU-Play OnlineB	U.S. ArmyBUbisoftBUbisoft EntertainmentBUnknown Worlds EntertainmentBUnreal SoftwareBVRChat Inc.BValveBVersus EvilB"Versus Evil;Obsidian EntertainmentBVertigo Gaming Inc.BVirtual Basement LLCBVlambeerBVostok GamesBWB GamesBWales InteractiveBWarchest Ltd.BWargaming Group LimitedBWarhorse Studios;Deep SilverB%Warner Bros Interactive EntertainmentB&Warner Bros. Interactive EntertainmentB>Warner Bros. Interactive Entertainment;Feral Interactive (Mac)BXWarner Bros. Interactive Entertainment;Feral Interactive (Mac);Feral Interactive (Linux)BWinged CloudBWizards of the Coast LLCBWube Software LTD.BX.D. Network Inc.B	YETU GAMEBYacht Club GamesBYoung HorsesBYsbryd Games;AGM PLAYISMBZero Point SoftwareBZombie Panic TeamBinXile EntertainmentBkChamp GamesBmarbenxBoddonegamesBthe whale husbandB	tinyBuildBtobyfoxB*北京网元圣唐娱乐科技有限公司B灵异调查管理局
?
Const_18Const*
_output_shapes	
:?*
dtype0	*?
value?B?	?"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?
Const_19Const*
_output_shapes
:*
dtype0*K
valueBB@BwindowsBwindows;linuxBwindows;macBwindows;mac;linux
q
Const_20Const*
_output_shapes
:*
dtype0	*5
value,B*	"                             
ά
Const_21Const*
_output_shapes	
:?*
dtype0*??
value??B???BeCross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;In-App Purchases;Steam LeaderboardsB?Local Multi-Player;Co-op;Local Co-op;Steam Achievements;Full controller support;VR Support;Steam Workshop;Stats;Steam LeaderboardsBMMOBMMO;Steam Trading CardsB\MMO;Steam Trading Cards;In-App Purchases;Partial Controller Support;Valve Anti-Cheat enabledB2MMO;Steam Trading Cards;Partial Controller SupportBMulti-playerBMulti-player;Co-opB-Multi-player;Co-op;Cross-Platform MultiplayerB?Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Captions available;VR Support;Partial Controller Support;Valve Anti-Cheat enabled;Includes level editor;Includes Source SDKB?Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Valve Anti-Cheat enabled;Includes level editorB?Multi-player;Co-op;Mods;Mods (require HL2);Steam Achievements;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Includes level editor;Includes Source SDKBLMulti-player;Co-op;Steam Achievements;Steam Trading Cards;Captions availableB?Multi-player;Co-op;Steam Achievements;Steam Trading Cards;StatsB^Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsBBMulti-player;Cross-Platform Multiplayer;Partial Controller SupportBWMulti-player;Cross-Platform Multiplayer;Steam Achievements;In-App Purchases;Steam CloudB?Multi-player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;Steam LeaderboardsB{Multi-player;Cross-Platform Multiplayer;Steam Trading Cards;Steam Workshop;Partial Controller Support;Includes level editorB\Multi-player;Cross-Platform Multiplayer;VR Support;Includes level editor;Includes Source SDKBJMulti-player;MMO;Co-op;Cross-Platform Multiplayer;Steam Achievements;StatsBRMulti-player;MMO;Co-op;Steam Achievements;Full controller support;In-App PurchasesBcMulti-player;MMO;Co-op;Steam Achievements;Steam Trading Cards;VR Support;Partial Controller SupportBQMulti-player;MMO;Co-op;Steam Trading Cards;In-App Purchases;Includes level editorBEMulti-player;MMO;Co-op;Steam Trading Cards;Partial Controller SupportB#Multi-player;MMO;Steam AchievementsB;Multi-player;MMO;Steam Achievements;Full controller supportB7Multi-player;MMO;Steam Achievements;Steam Trading CardsBMMulti-player;MMO;Steam Achievements;Steam Trading Cards;Includes level editorB$Multi-player;MMO;Steam Trading CardsB Multi-player;Online Multi-PlayerB?Multi-player;Online Multi-Player;Co-op;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Partial Controller Support;Steam CloudBLMulti-player;Online Multi-Player;Co-op;Online Co-op;Stats;Steam LeaderboardsB~Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBMulti-player;Online Multi-Player;Co-op;Online Co-op;Steam Trading Cards;In-App Purchases;Partial Controller Support;Steam CloudB}Multi-player;Online Multi-Player;Co-op;Steam Achievements;Steam Trading Cards;In-App Purchases;Valve Anti-Cheat enabled;StatsBgMulti-player;Online Multi-Player;Cross-Platform Multiplayer;In-App Purchases;Partial Controller SupportB?Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;Steam LeaderboardsB?Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Cloud;Valve Anti-Cheat enabledBIMulti-player;Online Multi-Player;Full controller support;In-App PurchasesB1Multi-player;Online Multi-Player;In-App PurchasesByMulti-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Partial Controller Support;Steam CloudB?Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Multi-player;Online Multi-Player;Local Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Workshop;Stats;Steam Leaderboards;Includes level editor;Includes Source SDKBbMulti-player;Online Multi-Player;Local Multi-Player;Shared/Split Screen;Partial Controller SupportBrMulti-player;Online Multi-Player;Local Multi-Player;Shared/Split Screen;Steam Achievements;Full controller supportB$Multi-player;Online Multi-Player;MMOB7Multi-player;Online Multi-Player;MMO;Co-op;Online Co-opB?Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;In-App Purchases;Valve Anti-Cheat enabled;StatsB?Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;In-App PurchasesBoMulti-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;In-App PurchasesB?Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;In-App Purchases;Partial Controller Support;StatsBKMulti-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Trading CardsB\Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Trading Cards;In-App PurchasesBwMulti-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Trading Cards;In-App Purchases;Partial Controller SupportBBMulti-player;Online Multi-Player;MMO;Online Co-op;In-App PurchasesBiMulti-player;Online Multi-Player;MMO;Online Co-op;Steam Achievements;Steam Trading Cards;In-App PurchasesBEMulti-player;Online Multi-Player;MMO;Online Co-op;Steam Trading CardsBqMulti-player;Online Multi-Player;MMO;Online Co-op;Steam Trading Cards;In-App Purchases;Partial Controller SupportBMulti-player;Online Multi-Player;MMO;Steam Achievements;Steam Trading Cards;Partial Controller Support;Valve Anti-Cheat enabledBIMulti-player;Online Multi-Player;MMO;Steam Trading Cards;In-App PurchasesBHMulti-player;Online Multi-Player;Online Co-op;Partial Controller SupportB?Multi-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;StatsB;Multi-player;Online Multi-Player;Partial Controller SupportB&Multi-player;Online Multi-Player;StatsBxMulti-player;Online Multi-Player;Steam Achievements;Full controller support;Steam Trading Cards;Valve Anti-Cheat enabledBXMulti-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;In-App PurchasesBsMulti-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;In-App Purchases;Partial Controller SupportB{Multi-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;Partial Controller Support;Valve Anti-Cheat enabledB?Multi-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;Steam Workshop;In-App Purchases;Stats;Includes level editorB?Multi-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Includes level editorB?Multi-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;Steam Workshop;Valve Anti-Cheat enabled;Stats;Includes level editorB`Multi-player;Online Multi-Player;Steam Trading Cards;In-App Purchases;Partial Controller SupportBwMulti-player;Online Multi-Player;Steam Trading Cards;Steam Workshop;Partial Controller Support;Valve Anti-Cheat enabledBTMulti-player;Online Multi-Player;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabledB@Multi-player;Partial Controller Support;Valve Anti-Cheat enabledBPMulti-player;Steam Achievements;Full controller support;Stats;Steam LeaderboardsB?Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Valve Anti-Cheat enabled;Includes level editorB3Multi-player;Steam Achievements;Steam Trading CardsBDMulti-player;Steam Achievements;Steam Trading Cards;In-App PurchasesBXMulti-player;Steam Achievements;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsB^Multi-player;Steam Achievements;Steam Trading Cards;Steam Cloud;Valve Anti-Cheat enabled;StatsBcMulti-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;StatsB Multi-player;Steam Trading CardsBOnline Multi-PlayerB?Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;In-App Purchases;Partial Controller Support;Includes level editorB$Online Multi-Player;In-App PurchasesBjOnline Multi-Player;MMO;Co-op;Online Co-op;Steam Workshop;Partial Controller Support;Includes level editorBdOnline Multi-Player;MMO;Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;In-App PurchasesBSOnline Multi-Player;MMO;Online Co-op;Cross-Platform Multiplayer;Steam Trading CardsBdOnline Multi-Player;MMO;Online Co-op;Cross-Platform Multiplayer;Steam Trading Cards;In-App PurchasesBKOnline Multi-Player;MMO;Online Co-op;Steam Achievements;Steam Trading CardsB\Online Multi-Player;MMO;Online Co-op;Steam Achievements;Steam Trading Cards;In-App PurchasesBdOnline Multi-Player;MMO;Online Co-op;Steam Trading Cards;In-App Purchases;Partial Controller SupportBSOnline Multi-Player;MMO;Online Co-op;Steam Trading Cards;Partial Controller SupportB Online Multi-Player;Online Co-opB?Online Multi-Player;Online Co-op;Steam Achievements;Steam Workshop;Partial Controller Support;Steam Cloud;Stats;Includes level editor;Includes Source SDKBEOnline Multi-Player;Online Co-op;Steam Trading Cards;In-App PurchasesB.Online Multi-Player;Partial Controller SupportB&Online Multi-Player;Steam AchievementsB2Online Multi-Player;Steam Achievements;Steam CloudB8Online Multi-Player;Steam Trading Cards;In-App PurchasesBSingle-playerB;Single-player;Captions available;Partial Controller SupportB?Single-player;Co-op;In-App Purchases;Partial Controller SupportB?Single-player;Co-op;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Co-op;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Workshop;Partial Controller Support;Includes level editorBwSingle-player;Co-op;Online Co-op;Local Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBkSingle-player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBbSingle-player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;Partial Controller SupportBRSingle-player;Co-op;Steam Achievements;Full controller support;Steam Trading CardsBkSingle-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;StatsB^Single-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBqSingle-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsBwSingle-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Valve Anti-Cheat enabledBmSingle-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam CloudBASingle-player;Co-op;Steam Achievements;Partial Controller SupportBtSingle-player;Co-op;Steam Achievements;Steam Trading Cards;Captions available;Partial Controller Support;Steam CloudBaSingle-player;Co-op;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam CloudBLSingle-player;Co-op;Steam Achievements;Steam Trading Cards;Steam Cloud;StatsB8Single-player;Co-op;Steam Cloud;Valve Anti-Cheat enabledB'Single-player;Co-op;Steam Trading CardsB?Single-player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;In-App Purchases;Steam Cloud;Includes level editorB%Single-player;Full controller supportBDSingle-player;Full controller support;Captions available;Steam CloudB;Single-player;Full controller support;Includes level editorB@Single-player;Full controller support;Partial Controller SupportB1Single-player;Full controller support;Steam CloudBESingle-player;Full controller support;Steam Trading Cards;Steam CloudB9Single-player;In-App Purchases;Partial Controller SupportBlSingle-player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading CardsB?Single-player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Workshop;Steam Cloud;Stats;Includes level editorBxSingle-player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Local Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Workshop;Steam Cloud;Steam Leaderboards;Includes level editorBdSingle-player;Local Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Local Multi-Player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Local Multi-Player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsBkSingle-player;Local Multi-Player;Local Co-op;Steam Achievements;Full controller support;Steam Trading CardsB~Single-player;Local Multi-Player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam CloudBqSingle-player;MMO;Co-op;Cross-Platform Multiplayer;Steam Achievements;In-App Purchases;Partial Controller SupportB=Single-player;MMO;Cross-Platform Multiplayer;In-App PurchasesBSingle-player;Multi-playerBiSingle-player;Multi-player;Captions available;VR Support;Partial Controller Support;Includes level editorB Single-player;Multi-player;Co-opBdSingle-player;Multi-player;Co-op;Captions available;Partial Controller Support;Includes level editorBvSingle-player;Multi-player;Co-op;Cross-Platform Multiplayer;Full controller support;Steam Trading Cards;Steam WorkshopBNSingle-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam AchievementsB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Includes level editorB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Partial Controller Support;Steam Cloud;Stats;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Captions available;Steam CloudB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Captions available;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Includes level editorB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam CloudB}Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam CloudB?Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorBsSingle-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Workshop;Includes level editorBLSingle-player;Multi-player;Co-op;In-App Purchases;Partial Controller SupportB?Single-player;Multi-player;Co-op;Local Co-op;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Cloud;Valve Anti-Cheat enabledB}Single-player;Multi-player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;In-App PurchasesB;Single-player;Multi-player;Co-op;Partial Controller SupportBGSingle-player;Multi-player;Co-op;Partial Controller Support;Steam CloudB?Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam CloudB?Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;StatsB?Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Steam Leaderboards;Includes level editorB?Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Includes level editorB?Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB~Single-player;Multi-player;Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Trading Cards;Partial Controller SupportBbSingle-player;Multi-player;Co-op;Shared/Split Screen;Full controller support;Includes level editorB|Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;In-App Purchases;Steam CloudBsSingle-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading CardsB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Stats;Steam LeaderboardsBSingle-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Steam Leaderboards;Includes level editorB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Steam LeaderboardsBvSingle-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Steam Trading Cards;Partial Controller SupportB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Steam Trading Cards;Partial Controller Support;Stats;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Shared/Split Screen;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;Valve Anti-Cheat enabled;Steam LeaderboardsB3Single-player;Multi-player;Co-op;Steam AchievementsBKSingle-player;Multi-player;Co-op;Steam Achievements;Full controller supportBjSingle-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Cloud;Steam LeaderboardsB_Single-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading CardsB?Single-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Cloud;Steam LeaderboardsBkSingle-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsB~Single-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Valve Anti-Cheat enabled;Steam LeaderboardsB?Single-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Includes level editorB?Single-player;Multi-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Valve Anti-Cheat enabled;Steam LeaderboardsBNSingle-player;Multi-player;Co-op;Steam Achievements;Partial Controller SupportBmSingle-player;Multi-player;Co-op;Steam Achievements;Partial Controller Support;Steam Cloud;Steam LeaderboardsBgSingle-player;Multi-player;Co-op;Steam Achievements;Partial Controller Support;Valve Anti-Cheat enabledBXSingle-player;Multi-player;Co-op;Steam Achievements;Steam Cloud;Stats;Steam LeaderboardsBXSingle-player;Multi-player;Co-op;Steam Achievements;Steam Cloud;Valve Anti-Cheat enabledBnSingle-player;Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam CloudB?Single-player;Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;Includes level editorB?Single-player;Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;Stats;Steam LeaderboardsBlSingle-player;Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsBxSingle-player;Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorBhSingle-player;Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;StatsB?Single-player;Multi-player;Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Steam Leaderboards;Includes level editorBiSingle-player;Multi-player;Co-op;Steam Achievements;Steam Workshop;Partial Controller Support;Steam CloudB?Single-player;Multi-player;Co-op;Steam Trading Cards;Captions available;In-App Purchases;Partial Controller Support;Includes level editorBESingle-player;Multi-player;Co-op;Steam Trading Cards;In-App PurchasesB`Single-player;Multi-player;Co-op;Steam Trading Cards;In-App Purchases;Partial Controller SupportB@Single-player;Multi-player;Co-op;Steam Trading Cards;Steam CloudBJSingle-player;Multi-player;Co-op;Steam Workshop;Partial Controller SupportB;Single-player;Multi-player;Co-op;Steam Workshop;Steam CloudBRSingle-player;Multi-player;Co-op;Valve Anti-Cheat enabled;Stats;Steam LeaderboardsB?Single-player;Multi-player;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Cloud;Stats;Steam LeaderboardsBtSingle-player;Multi-player;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading CardsB?Single-player;Multi-player;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBwSingle-player;Multi-player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam CloudB+Single-player;Multi-player;In-App PurchasesB0Single-player;Multi-player;Includes level editorB9Single-player;Multi-player;Local Multi-Player;Steam CloudB$Single-player;Multi-player;MMO;Co-opB?Single-player;Multi-player;MMO;Co-op;Steam Achievements;Full controller support;VR Support;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Includes level editorBoSingle-player;Multi-player;MMO;Co-op;Steam Achievements;Steam Trading Cards;Captions available;In-App PurchasesBwSingle-player;Multi-player;MMO;Co-op;Steam Achievements;Steam Trading Cards;In-App Purchases;Partial Controller SupportBJSingle-player;Multi-player;MMO;Co-op;VR Support;Partial Controller SupportBJSingle-player;Multi-player;MMO;In-App Purchases;Partial Controller SupportB7Single-player;Multi-player;MMO;Valve Anti-Cheat enabledB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Cross-Platform Multiplayer;Full controller support;Steam Workshop;Steam Leaderboards;Includes level editorBwSingle-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Cross-Platform Multiplayer;Partial Controller SupportB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam WorkshopBySingle-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Full controller support;Steam Trading Cards;Steam CloudBTSingle-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam AchievementsB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Captions available;Steam Workshop;Partial Controller Support;Steam Cloud;Stats;Steam Leaderboards;Includes level editor;Includes Source SDKB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Captions available;Steam CloudBxSingle-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Stats;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam WorkshopB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;In-App Purchases;Valve Anti-Cheat enabled;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Valve Anti-Cheat enabled;Stats;Steam Leaderboards;Includes level editorBySingle-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;In-App PurchasesB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;In-App Purchases;Partial Controller Support;Stats;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;Partial Controller SupportBtSingle-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;In-App Purchases;Partial Controller Support;Steam Cloud;Stats;Steam Leaderboards;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Co-op;Online Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editor;Commentary availableB?Single-player;Multi-player;Online Multi-Player;Co-op;Steam Achievements;Steam Workshop;Partial Controller Support;Steam Cloud;Includes level editorBZSingle-player;Multi-player;Online Multi-Player;Cross-Platform Multiplayer;In-App PurchasesB?Single-player;Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;VR Support;In-App Purchases;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Captions available;Steam Workshop;Partial Controller Support;Steam Cloud;Includes level editor;Includes Source SDKB|Single-player;Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam CloudBnSingle-player;Multi-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Trading Cards;In-App PurchasesBKSingle-player;Multi-player;Online Multi-Player;In-App Purchases;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Partial Controller Support;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading CardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Steam Leaderboards;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;In-App Purchases;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Steam Leaderboards;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Steam Achievements;Full controller support;Steam Trading Cards;In-App Purchases;Steam Cloud;Valve Anti-Cheat enabledB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Steam Achievements;Steam Trading Cards;Captions available;Steam Workshop;Partial Controller Support;Steam Leaderboards;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Steam Achievements;Steam Trading Cards;Steam Cloud;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Local Co-op;Steam Trading Cards;Steam WorkshopB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Cloud;Includes level editorB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Cloud;StatsBiSingle-player;Multi-player;Online Multi-Player;Local Multi-Player;Online Co-op;Local Co-op;Steam WorkshopB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Steam Achievements;Full controller support;Steam Cloud;Stats;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;Local Multi-Player;Steam Achievements;Steam Trading Cards;Partial Controller Support;StatsB{Single-player;Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Cross-Platform Multiplayer;Partial Controller SupportB]Single-player;Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Full controller supportBVSingle-player;Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;In-App PurchasesB?Single-player;Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Achievements;Full controller support;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Achievements;Steam Workshop;Partial Controller Support;Steam Cloud;Valve Anti-Cheat enabled;Stats;Steam LeaderboardsBjSingle-player;Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Trading Cards;In-App PurchasesBySingle-player;Multi-player;Online Multi-Player;MMO;Co-op;Online Co-op;Steam Trading Cards;Steam Workshop;In-App PurchasesBpSingle-player;Multi-player;Online Multi-Player;MMO;Co-op;Steam Achievements;Steam Trading Cards;In-App PurchasesBWSingle-player;Multi-player;Online Multi-Player;MMO;Steam Trading Cards;In-App PurchasesB?Single-player;Multi-player;Online Multi-Player;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBwSingle-player;Multi-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;StatsBUSingle-player;Multi-player;Online Multi-Player;Partial Controller Support;Steam CloudB?Single-player;Multi-player;Online Multi-Player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBASingle-player;Multi-player;Online Multi-Player;Steam AchievementsB?Single-player;Multi-player;Online Multi-Player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Workshop;In-App Purchases;Steam Cloud;Valve Anti-Cheat enabled;Stats;Steam Leaderboards;Includes level editorB`Single-player;Multi-player;Online Multi-Player;Steam Achievements;Steam Cloud;Steam LeaderboardsB:Single-player;Multi-player;Online Multi-Player;Steam CloudBqSingle-player;Multi-player;Online Multi-Player;Steam Workshop;Partial Controller Support;Stats;Steam LeaderboardsB5Single-player;Multi-player;Partial Controller SupportB?Single-player;Multi-player;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Multi-player;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Valve Anti-Cheat enabled;Steam LeaderboardsB?Single-player;Multi-player;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Includes level editorBeSingle-player;Multi-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam CloudB-Single-player;Multi-player;Steam AchievementsBESingle-player;Multi-player;Steam Achievements;Full controller supportB^Single-player;Multi-player;Steam Achievements;Full controller support;Stats;Steam LeaderboardsBdSingle-player;Multi-player;Steam Achievements;Full controller support;Steam Cloud;Steam LeaderboardsB}Single-player;Multi-player;Steam Achievements;Full controller support;Steam Cloud;Valve Anti-Cheat enabled;Steam LeaderboardsBXSingle-player;Multi-player;Steam Achievements;Full controller support;Steam LeaderboardsB?Single-player;Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Cloud;Steam LeaderboardsBeSingle-player;Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBxSingle-player;Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Valve Anti-Cheat enabled;Steam LeaderboardsB~Single-player;Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Includes level editorB?Single-player;Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Valve Anti-Cheat enabled;Steam LeaderboardsBCSingle-player;Multi-player;Steam Achievements;Includes level editorBHSingle-player;Multi-player;Steam Achievements;Partial Controller SupportBmSingle-player;Multi-player;Steam Achievements;Partial Controller Support;Steam Cloud;Stats;Steam LeaderboardsBgSingle-player;Multi-player;Steam Achievements;Partial Controller Support;Steam Cloud;Steam LeaderboardsBmSingle-player;Multi-player;Steam Achievements;Partial Controller Support;Steam Cloud;Valve Anti-Cheat enabledBFSingle-player;Multi-player;Steam Achievements;Stats;Steam LeaderboardsB9Single-player;Multi-player;Steam Achievements;Steam CloudBASingle-player;Multi-player;Steam Achievements;Steam Trading CardsB?Single-player;Multi-player;Steam Achievements;Steam Trading Cards;Captions available;Steam Workshop;Steam Cloud;Stats;Steam Leaderboards;Includes level editorBoSingle-player;Multi-player;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam LeaderboardsBGSingle-player;Multi-player;Steam Achievements;Steam Trading Cards;StatsBMSingle-player;Multi-player;Steam Achievements;Steam Trading Cards;Steam CloudBSingle-player;Multi-player;Steam Achievements;Steam Trading Cards;Steam Cloud;Valve Anti-Cheat enabled;Stats;Steam LeaderboardsBPSingle-player;Multi-player;Steam Achievements;Steam Trading Cards;Steam WorkshopB}Single-player;Multi-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam Cloud;StatsB?Single-player;Multi-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam Cloud;Stats;Steam Leaderboards;Includes level editorB\Single-player;Multi-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam CloudB?Single-player;Multi-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Valve Anti-Cheat enabled;Stats;Includes level editorBzSingle-player;Multi-player;Steam Achievements;Steam Trading Cards;VR Support;Partial Controller Support;Steam LeaderboardsBHSingle-player;Multi-player;Steam Achievements;Steam Workshop;Steam CloudB[Single-player;Multi-player;Steam Achievements;Steam Workshop;Steam Cloud;Steam LeaderboardsB&Single-player;Multi-player;Steam CloudB.Single-player;Multi-player;Steam Trading CardsBZSingle-player;Multi-player;Steam Trading Cards;In-App Purchases;Partial Controller SupportBISingle-player;Multi-player;Steam Trading Cards;Partial Controller SupportB:Single-player;Multi-player;Steam Trading Cards;Steam CloudBPSingle-player;Multi-player;Steam Trading Cards;Steam Cloud;Includes level editorBISingle-player;Multi-player;Steam Trading Cards;Steam Workshop;Steam CloudBGSingle-player;Multi-player;Steam Trading Cards;Valve Anti-Cheat enabledB3Single-player;Multi-player;Valve Anti-Cheat enabledB2Single-player;Online Co-op;Full controller supportBtSingle-player;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam CloudB4Single-player;Online Multi-Player;Co-op;Online Co-opBeSingle-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Includes level editorBtSingle-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;In-App PurchasesBzSingle-player;Online Multi-Player;Cross-Platform Multiplayer;Steam Achievements;Steam Workshop;Stats;Includes level editorB>Single-player;Online Multi-Player;In-App Purchases;Steam CloudB?Single-player;Online Multi-Player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;StatsB?Single-player;Online Multi-Player;Local Multi-Player;Local Co-op;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsB[Single-player;Online Multi-Player;Local Multi-Player;Partial Controller Support;Steam CloudB?Single-player;Online Multi-Player;Local Multi-Player;Shared/Split Screen;Cross-Platform Multiplayer;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Includes level editorB?Single-player;Online Multi-Player;Local Multi-Player;Shared/Split Screen;Steam Achievements;Full controller support;Captions availableB^Single-player;Online Multi-Player;MMO;Online Co-op;In-App Purchases;Partial Controller SupportB?Single-player;Online Multi-Player;Online Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam Cloud;Stats;Steam Leaderboards;Includes level editorBeSingle-player;Online Multi-Player;Online Co-op;Full controller support;Steam Cloud;Steam LeaderboardsBmSingle-player;Online Multi-Player;Online Co-op;Shared/Split Screen;Steam Achievements;Full controller supportBYSingle-player;Online Multi-Player;Online Co-op;Steam Achievements;Full controller supportBeSingle-player;Online Multi-Player;Online Co-op;Steam Achievements;Full controller support;Steam CloudBySingle-player;Online Multi-Player;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudB?Single-player;Online Multi-Player;Online Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsBrSingle-player;Online Multi-Player;Online Co-op;Steam Achievements;Full controller support;Valve Anti-Cheat enabledBMSingle-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam CloudBUSingle-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam Trading CardsBSingle-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller SupportB?Single-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Includes level editorB?Single-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Online Multi-Player;Online Co-op;Steam Achievements;Steam Trading Cards;VR Support;Steam Workshop;In-App Purchases;Steam Cloud;Valve Anti-Cheat enabled;Stats;Includes level editorBnSingle-player;Online Multi-Player;Online Co-op;Steam Trading Cards;In-App Purchases;Partial Controller SupportB?Single-player;Online Multi-Player;Steam Achievements;Full controller support;Steam Trading Cards;In-App Purchases;Steam Cloud;Valve Anti-Cheat enabledBlSingle-player;Online Multi-Player;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBtSingle-player;Online Multi-Player;Steam Achievements;Partial Controller Support;Steam Cloud;Stats;Steam LeaderboardsBxSingle-player;Online Multi-Player;Steam Achievements;Partial Controller Support;Steam Leaderboards;Includes level editorBNSingle-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;StatsBSingle-player;Online Multi-Player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Includes level editorBaSingle-player;Online Multi-Player;Steam Trading Cards;In-App Purchases;Partial Controller SupportB(Single-player;Partial Controller SupportB=Single-player;Partial Controller Support;Commentary availableB4Single-player;Partial Controller Support;Steam CloudB9Single-player;Shared/Split Screen;Full controller supportB<Single-player;Shared/Split Screen;Partial Controller SupportBXSingle-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam CloudBqSingle-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Cloud;Stats;Steam LeaderboardsB`Single-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading CardsBlSingle-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBSingle-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB?Single-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Steam LeaderboardsB?Single-player;Shared/Split Screen;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Steam Leaderboards;Includes level editorBOSingle-player;Shared/Split Screen;Steam Achievements;Partial Controller SupportB?Single-player;Shared/Split Screen;Steam Achievements;Steam Trading Cards;In-App Purchases;Partial Controller Support;Steam Cloud;Steam LeaderboardsB Single-player;Steam AchievementsBZSingle-player;Steam Achievements;Captions available;Partial Controller Support;Steam CloudBeSingle-player;Steam Achievements;Captions available;VR Support;Partial Controller Support;Steam CloudB8Single-player;Steam Achievements;Full controller supportBvSingle-player;Steam Achievements;Full controller support;Captions available;Includes level editor;Commentary availableBWSingle-player;Steam Achievements;Full controller support;Captions available;Steam CloudB[Single-player;Steam Achievements;Full controller support;In-App Purchases;Steam Cloud;StatsBDSingle-player;Steam Achievements;Full controller support;Steam CloudBYSingle-player;Steam Achievements;Full controller support;Steam Cloud;Commentary availableBJSingle-player;Steam Achievements;Full controller support;Steam Cloud;StatsB]Single-player;Steam Achievements;Full controller support;Steam Cloud;Stats;Steam LeaderboardsBWSingle-player;Steam Achievements;Full controller support;Steam Cloud;Steam LeaderboardsBKSingle-player;Steam Achievements;Full controller support;Steam LeaderboardsBLSingle-player;Steam Achievements;Full controller support;Steam Trading CardsB_Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Captions availableBkSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam CloudB?Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Cloud;Commentary availableB?Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Cloud;Includes level editorB?Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Cloud;Stats;Steam LeaderboardsB~Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Cloud;Steam LeaderboardsBiSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;In-App Purchases;Steam CloudBXSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam CloudBmSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Commentary availableB^Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;StatsBkSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam LeaderboardsB_Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam LeaderboardsB[Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam WorkshopB?Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;In-App Purchases;Steam Cloud;Steam LeaderboardsBgSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam CloudB}Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB?Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Steam LeaderboardsB?Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Steam Leaderboards;Includes level editorBzSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;Steam Cloud;Steam LeaderboardsBWSingle-player;Steam Achievements;Full controller support;Steam Trading Cards;VR SupportBvSingle-player;Steam Achievements;Full controller support;Steam Workshop;Stats;Steam Leaderboards;Includes level editorBSSingle-player;Steam Achievements;Full controller support;Steam Workshop;Steam CloudBpSingle-player;Steam Achievements;Full controller support;Steam Workshop;Steam Leaderboards;Includes level editorB;Single-player;Steam Achievements;Partial Controller SupportBTSingle-player;Steam Achievements;Partial Controller Support;Stats;Steam LeaderboardsBGSingle-player;Steam Achievements;Partial Controller Support;Steam CloudB`Single-player;Steam Achievements;Partial Controller Support;Steam Cloud;Stats;Steam LeaderboardsBZSingle-player;Steam Achievements;Partial Controller Support;Steam Cloud;Steam LeaderboardsBNSingle-player;Steam Achievements;Partial Controller Support;Steam LeaderboardsB,Single-player;Steam Achievements;Steam CloudBESingle-player;Steam Achievements;Steam Cloud;Stats;Steam LeaderboardsB3Single-player;Steam Achievements;Steam LeaderboardsB4Single-player;Steam Achievements;Steam Trading CardsBsSingle-player;Steam Achievements;Steam Trading Cards;Captions available;In-App Purchases;Partial Controller SupportBwSingle-player;Steam Achievements;Steam Trading Cards;Captions available;Partial Controller Support;Commentary availableB?Single-player;Steam Achievements;Steam Trading Cards;Captions available;Partial Controller Support;Steam Cloud;Commentary availableBSSingle-player;Steam Achievements;Steam Trading Cards;Captions available;Steam CloudBiSingle-player;Steam Achievements;Steam Trading Cards;Captions available;Steam Cloud;Includes level editorBxSingle-player;Steam Achievements;Steam Trading Cards;Captions available;Steam Workshop;Steam Cloud;Includes level editorB^Single-player;Steam Achievements;Steam Trading Cards;In-App Purchases;Stats;Steam LeaderboardsBWSingle-player;Steam Achievements;Steam Trading Cards;In-App Purchases;Steam Cloud;StatsBOSingle-player;Steam Achievements;Steam Trading Cards;Partial Controller SupportBeSingle-player;Steam Achievements;Steam Trading Cards;Partial Controller Support;Includes level editorB[Single-player;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam CloudBtSingle-player;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;Stats;Steam LeaderboardsBnSingle-player;Steam Achievements;Steam Trading Cards;Partial Controller Support;Steam Cloud;Steam LeaderboardsB:Single-player;Steam Achievements;Steam Trading Cards;StatsB@Single-player;Steam Achievements;Steam Trading Cards;Steam CloudBUSingle-player;Steam Achievements;Steam Trading Cards;Steam Cloud;Commentary availableBYSingle-player;Steam Achievements;Steam Trading Cards;Steam Cloud;Stats;Steam LeaderboardsBSSingle-player;Steam Achievements;Steam Trading Cards;Steam Cloud;Steam LeaderboardsBGSingle-player;Steam Achievements;Steam Trading Cards;Steam LeaderboardsBCSingle-player;Steam Achievements;Steam Trading Cards;Steam WorkshopBTSingle-player;Steam Achievements;Steam Trading Cards;Steam Workshop;In-App PurchasesBjSingle-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam CloudB?Single-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam Cloud;Includes level editorB?Single-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam Cloud;Stats;Includes level editorBOSingle-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam CloudBeSingle-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB~Single-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Stats;Steam Leaderboards;Includes level editorBbSingle-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam Cloud;Steam LeaderboardsBVSingle-player;Steam Achievements;Steam Trading Cards;Steam Workshop;Steam LeaderboardsBfSingle-player;Steam Achievements;Steam Trading Cards;VR Support;Partial Controller Support;Steam CloudB/Single-player;Steam Achievements;Steam WorkshopBESingle-player;Steam Achievements;Steam Workshop;Includes level editorBVSingle-player;Steam Achievements;Steam Workshop;Partial Controller Support;Steam CloudBSingle-player;Steam CloudB Single-player;Steam LeaderboardsB!Single-player;Steam Trading CardsB<Single-player;Steam Trading Cards;Partial Controller SupportB-Single-player;Steam Trading Cards;Steam CloudBWSingle-player;Steam Trading Cards;Steam Workshop;Partial Controller Support;Steam CloudBRSingle-player;Steam Trading Cards;Steam Workshop;Steam Cloud;Includes level editorB2Single-player;Steam Workshop;Includes level editorBMSingle-player;Steam Workshop;Partial Controller Support;Includes level editorB(Single-player;Steam Workshop;Steam CloudB&Steam Achievements;Steam Trading CardsBKSteam Achievements;Steam Trading Cards;Steam Workshop;Includes level editorBSteam Workshop
?
Const_22Const*
_output_shapes	
:?*
dtype0	*?
value?B?	?"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?>
Const_23Const*
_output_shapes	
:?*
dtype0*?>
value?>B?>?BActionBAction;AdventureBAction;Adventure;CasualB*Action;Adventure;Casual;Free to Play;IndieBDAction;Adventure;Casual;Free to Play;Indie;Massively Multiplayer;RPGBOAction;Adventure;Casual;Free to Play;Indie;Massively Multiplayer;RPG;SimulationB>Action;Adventure;Casual;Free to Play;Massively Multiplayer;RPGBIAction;Adventure;Casual;Free to Play;Massively Multiplayer;RPG;SimulationBAction;Adventure;Casual;IndieB@Action;Adventure;Casual;Indie;Massively Multiplayer;Early AccessB!Action;Adventure;Casual;Indie;RPGB(Action;Adventure;Casual;Indie;SimulationB&Action;Adventure;Casual;Indie;StrategyB-Action;Adventure;Casual;Massively MultiplayerBAction;Adventure;Free to PlayB#Action;Adventure;Free to Play;IndieBFAction;Adventure;Free to Play;Indie;Massively Multiplayer;Early AccessB=Action;Adventure;Free to Play;Indie;Massively Multiplayer;RPGB3Action;Adventure;Free to Play;Massively MultiplayerB7Action;Adventure;Free to Play;Massively Multiplayer;RPGB/Action;Adventure;Free to Play;Simulation;SportsB-Action;Adventure;Free to Play;Sports;StrategyBAction;Adventure;IndieB#Action;Adventure;Indie;Early AccessB9Action;Adventure;Indie;Massively Multiplayer;Early AccessB0Action;Adventure;Indie;Massively Multiplayer;RPGBDAction;Adventure;Indie;Massively Multiplayer;RPG;Simulation;StrategyBDAction;Adventure;Indie;Massively Multiplayer;Simulation;Early AccessBAction;Adventure;Indie;RPGB'Action;Adventure;Indie;RPG;Early AccessB%Action;Adventure;Indie;RPG;SimulationB(Action;Adventure;Indie;Racing;SimulationB!Action;Adventure;Indie;SimulationB.Action;Adventure;Indie;Simulation;Early AccessB*Action;Adventure;Indie;Simulation;StrategyB&Action;Adventure;Massively MultiplayerB7Action;Adventure;Massively Multiplayer;RPG;Early AccessBBAction;Adventure;Massively Multiplayer;RPG;Simulation;Early AccessB>Action;Adventure;Massively Multiplayer;RPG;Simulation;StrategyBAction;Adventure;RPGBAction;Adventure;RacingB(Action;Adventure;Simulation;Early AccessBAction;Adventure;StrategyB Action;Casual;Free to Play;IndieB+Action;Casual;Free to Play;Indie;SimulationB0Action;Casual;Free to Play;Massively MultiplayerB7Action;Casual;Free to Play;Massively Multiplayer;SportsBAction;Casual;IndieB?Action;Casual;Indie;Massively Multiplayer;Strategy;Early AccessB!Action;Casual;Indie;Racing;SportsBAction;Casual;Indie;SimulationBAction;Casual;Indie;StrategyB)Action;Casual;Indie;Strategy;Early AccessBAction;Early AccessBAction;Free to PlayB Action;Free to Play;Early AccessBAction;Free to Play;IndieB&Action;Free to Play;Indie;Early AccessB/Action;Free to Play;Indie;Massively MultiplayerB3Action;Free to Play;Indie;Massively Multiplayer;RPGB@Action;Free to Play;Indie;Massively Multiplayer;RPG;Early AccessB>Action;Free to Play;Indie;Massively Multiplayer;RPG;SimulationBAction;Free to Play;Indie;RPGB&Action;Free to Play;Indie;RPG;StrategyB-Action;Free to Play;Indie;Simulation;StrategyB"Action;Free to Play;Indie;StrategyB/Action;Free to Play;Indie;Strategy;Early AccessB)Action;Free to Play;Massively MultiplayerB6Action;Free to Play;Massively Multiplayer;Early AccessB-Action;Free to Play;Massively Multiplayer;RPGB:Action;Free to Play;Massively Multiplayer;RPG;Early AccessB6Action;Free to Play;Massively Multiplayer;RPG;StrategyB0Action;Free to Play;Massively Multiplayer;RacingB4Action;Free to Play;Massively Multiplayer;SimulationB=Action;Free to Play;Massively Multiplayer;Simulation;StrategyB2Action;Free to Play;Massively Multiplayer;StrategyBAction;Free to Play;StrategyBAction;IndieBAction;Indie;Early AccessB1Action;Indie;Massively Multiplayer;RPG;SimulationB<Action;Indie;Massively Multiplayer;RPG;Strategy;Early AccessBAction;Indie;RPGBAction;Indie;RPG;SimulationB(Action;Indie;RPG;Simulation;Early AccessB$Action;Indie;RPG;Simulation;StrategyBAction;Indie;RPG;StrategyB&Action;Indie;RPG;Strategy;Early AccessBAction;Indie;RacingB%Action;Indie;Racing;Simulation;SportsBAction;Indie;Racing;SportsBAction;Indie;SimulationB Action;Indie;Simulation;StrategyBAction;Indie;SportsBAction;Indie;StrategyB"Action;Indie;Strategy;Early AccessB#Action;Massively Multiplayer;RacingB'Action;Massively Multiplayer;SimulationB0Action;Massively Multiplayer;Simulation;StrategyB
Action;RPGBAction;RPG;StrategyBAction;SimulationBAction;Simulation;StrategyBAction;SportsBAction;StrategyB	AdventureB#Adventure;Casual;Free to Play;IndieBFAdventure;Casual;Free to Play;Indie;Massively Multiplayer;RPG;StrategyB;Adventure;Casual;Free to Play;Indie;RPG;Simulation;StrategyB.Adventure;Casual;Free to Play;Indie;SimulationB7Adventure;Casual;Free to Play;Massively Multiplayer;RPGBRAdventure;Casual;Free to Play;Massively Multiplayer;Simulation;Sports;Early AccessBAdventure;Casual;IndieB%Adventure;Casual;Indie;RPG;SimulationB0Adventure;Casual;Indie;RPG;Strategy;Early AccessB(Adventure;Casual;Indie;Racing;SimulationB!Adventure;Casual;Indie;SimulationB*Adventure;Casual;Indie;Simulation;StrategyBAdventure;Casual;StrategyBAdventure;Free to PlayBAdventure;Free to Play;IndieB Adventure;Free to Play;Indie;RPGB0Adventure;Free to Play;Massively Multiplayer;RPGB9Adventure;Free to Play;Massively Multiplayer;RPG;StrategyBAdventure;IndieBAdventure;Indie;Early AccessB2Adventure;Indie;Massively Multiplayer;RPG;StrategyBAdventure;Indie;RPGBAdventure;Indie;RPG;SimulationBAdventure;Indie;RPG;StrategyBAdventure;Indie;SimulationB'Adventure;Indie;Simulation;Early AccessB#Adventure;Indie;Simulation;StrategyB0Adventure;Indie;Simulation;Strategy;Early AccessBAdventure;Indie;StrategyBAdventure;RPGBAdventure;RPG;StrategyBAdventure;Simulation;SportsBAdventure;Simulation;StrategyBAdventure;StrategyB4Animation & Modeling;Design & Illustration;UtilitiesB%Animation & Modeling;Video ProductionBCasual;Free to Play;IndieB8Casual;Free to Play;Indie;Massively Multiplayer;StrategyB$Casual;Free to Play;Indie;SimulationBCasual;IndieBCasual;Indie;RPG;SimulationB$Casual;Indie;RPG;Simulation;StrategyBCasual;Indie;SimulationB$Casual;Indie;Simulation;Early AccessB Casual;Indie;Simulation;StrategyB-Casual;Indie;Simulation;Strategy;Early AccessB Casual;Indie;Sports;Early AccessBCasual;Indie;StrategyBCasual;SimulationBFree to PlayBFree to Play;IndieB(Free to Play;Indie;Massively MultiplayerBFree to Play;Indie;SimulationB$Free to Play;Indie;Simulation;SportsB&Free to Play;Massively Multiplayer;RPGB4Free to Play;Massively Multiplayer;Simulation;SportsB+Free to Play;Massively Multiplayer;StrategyBFree to Play;RPG;SimulationB%Free to Play;Racing;Simulation;SportsBFree to Play;SimulationBFree to Play;StrategyBIndieBIndie;Early AccessBIndie;Massively Multiplayer;RPGB	Indie;RPGBIndie;RPG;SimulationB!Indie;RPG;Simulation;Early AccessBIndie;RPG;Simulation;StrategyBIndie;RPG;StrategyB$Indie;Racing;Simulation;Early AccessBIndie;Racing;Simulation;SportsBIndie;SimulationBIndie;Simulation;Early AccessBIndie;Simulation;SportsBIndie;Simulation;StrategyBIndie;StrategyBMassively Multiplayer;RPGBNudity;Adventure;Casual;IndieBONudity;Violent;Gore;Action;Adventure;Indie;RPG;Simulation;Strategy;Early AccessBRPGB	RPG;IndieBRPG;Simulation;StrategyBRPG;StrategyBRacingBRacing;SimulationBRacing;Simulation;Early AccessBRacing;Simulation;SportsB!Racing;Simulation;Sports;StrategyBRacing;SportsB/Sexual Content;Nudity;Casual;Free to Play;IndieB"Sexual Content;Nudity;Casual;IndieBSexual Content;Nudity;IndieB3Sexual Content;Nudity;Violent;Gore;Action;AdventureB9Sexual Content;Nudity;Violent;Gore;Action;Adventure;IndieB&Sexual Content;Nudity;Violent;Gore;RPGB/Sexual Content;Violent;Adventure;Indie;StrategyB
SimulationBSimulation;SportsBSimulation;StrategyBSportsBStrategyBStrategy;Early AccessB	UtilitiesBViolent;Action;AdventureB;Violent;Action;Adventure;Free to Play;Massively MultiplayerBViolent;Action;Adventure;IndieBViolent;Action;StrategyB0Violent;Free to Play;Indie;Strategy;Early AccessB#Violent;Gore;Action;Adventure;IndieBViolent;Gore;Action;Indie;RPGB'Violent;Gore;Action;Simulation;Strategy
?
Const_24Const*
_output_shapes	
:?*
dtype0	*?
value?B?	?"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       
ޯ
Const_25Const*
_output_shapes	
:?*
dtype0*??
value??B???B%3D Platformer;Cute;Female ProtagonistBAction RPG;Pixel Graphics;RPGBAction RPG;RPG;Hack and SlashBAction;Adventure;Hack and SlashBAction;Adventure;MultiplayerBAction;Adventure;Open WorldBAction;Batman;Open WorldBAction;Batman;StealthBAction;Comedy;AdventureBAction;FPS;AliensBAction;FPS;Co-opBAction;FPS;GoreBAction;FPS;MultiplayerBAction;FPS;Open WorldBAction;FPS;Sci-fiBAction;FPS;SwordplayBAction;FPS;World War IIBAction;FPS;ZombiesBAction;Fast-Paced;IndieB(Action;Female Protagonist;Hack and SlashBAction;Gore;FPSB+Action;Hack and Slash;Character Action GameB(Action;Hack and Slash;Female ProtagonistB&Action;Hack and Slash;Great SoundtrackBAction;Hack and Slash;SurrealBAction;Horror;AdventureBAction;Horror;ZombiesBAction;Hunting;MultiplayerBAction;Multiplayer;FPSBAction;Multiplayer;GoreB'Action;Multiplayer;Third-Person ShooterBAction;Online Co-Op;Co-opBAction;Open World;AdventureBAction;Open World;BatmanBAction;Open World;ComedyBAction;Open World;CrimeBAction;Open World;GoreBAction;Open World;ParkourB"Action;Platformer;Great SoundtrackBAction;Platformer;IndieBAction;Souls-like;DifficultBAction;Souls-like;RPGBAction;Story Rich;Beat 'em upB#Action;Story Rich;Time ManipulationBAction;Tactical;Sci-fiBAction;Tactical;ShooterB'Action;Third-Person Shooter;Bullet TimeB!Action;Third-Person Shooter;Co-opB"Action;Third-Person Shooter;Sci-fiBAction;Zombies;Co-opB#Adventure;Action;Female ProtagonistBAdventure;Action;Lara CroftB Adventure;Atmospheric;Story RichBAdventure;Classic;Point & ClickBAdventure;Detective;Story RichB"Adventure;First-Person;ExplorationBAdventure;First-Person;ParkourBAdventure;Mythology;ActionBAdventure;Open World;SandboxB#Adventure;Point & Click;AtmosphericBAdventure;Point & Click;ComedyBAdventure;RPG;ActionBAmerica;Action;Pixel GraphicsB2Animation & Modeling;Video Production;Free to PlayBAnime;Action;AdventureBAnime;Board Game;CuteBAnime;Cute;Bullet HellBAnime;Detective;Visual NovelBAnime;Fighting;ActionBAnime;Fighting;NinjaBAnime;Free to Play;MMORPGBAnime;Free to Play;RPGBAnime;JRPG;Female ProtagonistBAnime;JRPG;RPGBAnime;Mystery;Visual NovelBAnime;Nudity;MatureBAnime;Nudity;Sexual ContentB!Anime;RPG;Character CustomizationBAnime;Sexual Content;CuteBAnime;Sexual Content;NudityBAnime;Strategy;Turn-BasedBAnime;Visual Novel;NudityBArcade;Classic;ActionBArcade;Retro;ClassicBAssassin;Open World;ActionBAtmospheric;Dark;IndieB&Atmospheric;Great Soundtrack;BeautifulB'Atmospheric;Great Soundtrack;PlatformerB"Atmospheric;Historical;World War IB'Atmospheric;Post-apocalyptic;Open WorldB$Atmospheric;Psychological;Story RichBBase-Building;Survival;StrategyBBasketball;Sports;MultiplayerBBatman;Action;Open WorldBBatman;Superhero;Story RichB!Battle Royale;Action;Free to PlayB!Board Game;Simulation;MultiplayerB'Board Game;Turn-Based Strategy;StrategyBBuilding;Dinosaurs;ManagementBBuilding;Open World;SandboxBBuilding;Simulation;PhysicsB&Bullet Hell;Great Soundtrack;DifficultB%Bullet Hell;Pixel Graphics;Rogue-likeBCapitalism;Anime;RPGBCard Game;RPG;ActionBCard Game;Strategy;MultiplayerB$Card Game;Trading Card Game;StrategyBCard Game;Turn-Based;Rogue-likeBCasual;Action;IndieBCasual;Indie;PuzzleB$Choices Matter;Dystopian ;Story RichB,Choices Matter;FMV;Choose Your Own AdventureBChoices Matter;Indie;MedievalB#Choices Matter;Story Rich;AdventureB City Builder;Base-Building;SpaceBCity Builder;Building;SandboxB City Builder;Simulation;BuildingB City Builder;Simulation;StrategyBCity Builder;Strategy;MedievalB City Builder;Strategy;SimulationBCity Builder;Survival;StrategyBClicker;Free to Play;CasualBCo-op;Action;AdventureBCo-op;Action;FPSBCo-op;Dungeon Crawler;ActionBCo-op;Gore;First-PersonB%Co-op;Online Co-Op;Twin Stick ShooterBCo-op;Puzzle;Local Co-OpBCo-op;Stealth;IndieBComedy;Action;Co-opBComedy;Adventure;Story RichBComedy;Co-op;PlatformerBComedy;Funny;SportsBComedy;Indie;AdventureBComedy;Narration;IndieBComedy;Physics;IndieBCrime;Open World;ActionBCult Classic;Competitive;RacingBCult Classic;Physics;IndieBCute;Exploration;AdventureB#Cyberpunk;Action;Female ProtagonistB!Cyberpunk;Action;Great SoundtrackBCyberpunk;RPG;FPSBCyberpunk;RPG;StealthBCyberpunk;Stealth;ActionBCyberpunk;Stealth;RPGB'Cyberpunk;Visual Novel;Great SoundtrackBDark Fantasy;Difficult;ActionBDark Fantasy;Difficult;RPGBDark Humor;RPG;Story RichBDark Humor;Violent;ActionBDetective;Adventure;HorrorBDetective;Crime;Open WorldBDetective;FMV;IndieBDetective;Indie;RPGBDifficult;Cartoon;PlatformerB Difficult;Great Soundtrack;MusicB Difficult;Indie;Great SoundtrackB&Difficult;Psychological Horror;PhysicsBDinosaurs;Action;FPSBDinosaurs;Action;MultiplayerBDinosaurs;Multiplayer;ActionBDog;FPS;ActionB+Dungeon Crawler;Hack and Slash;SingleplayerBDungeon Crawler;RPG;IndieBEarly Access;Action;FPSBEarly Access;Action;GoreB.Early Access;Base-Building;Resource ManagementB#Early Access;Base-Building;StrategyB#Early Access;Base-Building;SurvivalB'Early Access;Battle Royale;Free to PlayB#Early Access;Battle Royale;SurvivalB!Early Access;Building;MultiplayerBEarly Access;Building;SandboxBEarly Access;Co-op;ActionBEarly Access;Dinosaurs;SurvivalBEarly Access;FPS;Arena ShooterBEarly Access;FPS;MilitaryBEarly Access;FPS;MultiplayerBEarly Access;FPS;RemakeBEarly Access;FPS;WarB'Early Access;Free to Play;Battle RoyaleBEarly Access;Free to Play;FPSBEarly Access;Free to Play;MemesB!Early Access;Free to Play;ShooterB"Early Access;Free to Play;StrategyB"Early Access;Free to Play;SurvivalB!Early Access;Free to Play;ZombiesB#Early Access;Historical;MultiplayerBEarly Access;Indie;ActionB,Early Access;Mini Golf;Massively MultiplayerB"Early Access;Mini Golf;MultiplayerB!Early Access;Multiplayer;SurvivalB Early Access;Open World;SurvivalB Early Access;Pirates;MultiplayerB&Early Access;Pixel Graphics;Rogue-likeBEarly Access;RPG;SandboxB#Early Access;Rogue-like;MultiplayerBEarly Access;Simulation;DrivingBEarly Access;Space;SandboxBEarly Access;Strategy;Card GameBEarly Access;Strategy;FPSB+Early Access;Strategy;Massively MultiplayerB!Early Access;Survival;MultiplayerB Early Access;Survival;Open WorldBEarly Access;Survival;SpaceBEarly Access;Survival;ZombiesBEarly Access;Tanks;MultiplayerBEarly Access;VR;RhythmB!Exploration;Survival;LovecraftianBFPS;1980s;ComedyBFPS;Action;Bullet TimeBFPS;Action;Co-opBFPS;Action;GoreBFPS;Action;MultiplayerBFPS;Action;NudityBFPS;Action;Sci-fiBFPS;Action;ShooterBFPS;Action;Story RichBFPS;Co-op;RPGBFPS;Horror;Co-opBFPS;Indie;DifficultBFPS;Multiplayer;ActionBFPS;Multiplayer;Arena ShooterBFPS;Multiplayer;TacticalBFPS;Post-apocalyptic;ActionBFPS;Realistic;ShooterBFPS;Realistic;TacticalBFPS;Shooter;ActionBFPS;Story Rich;ActionBFPS;War;MultiplayerBFemale Protagonist;Noir;IndieBFighting;2D Fighter;ArcadeBFighting;2D Fighter;IndieBFighting;2D Fighter;MultiplayerBFighting;Action;Martial ArtsBFighting;Action;MultiplayerBFighting;Anime;ActionBFighting;Arcade;CompetitiveB"Fighting;Female Protagonist;ActionBFighting;Gore;ActionB%Fighting;Local Multiplayer;2D FighterB Fighting;Multiplayer;CompetitiveBFighting;Superhero;ActionBFlight;Action;Story RichB&Free to Play;Action RPG;Hack and SlashBFree to Play;Action;BuildingBFree to Play;Action;Co-opBFree to Play;Action;FightingBFree to Play;Action;MultiplayerBFree to Play;Action;Open WorldBFree to Play;Action;ShooterBFree to Play;Anime;ActionBFree to Play;Anime;RPGBFree to Play;Anime;Visual NovelBFree to Play;Basketball;SportsB Free to Play;Board Game;StrategyBFree to Play;Card Game;AnimeB"Free to Play;Card Game;MultiplayerB(Free to Play;Card Game;Trading Card GameBFree to Play;Clicker;CapitalismBFree to Play;Co-op;MultiplayerB Free to Play;Comedy;First-PersonBFree to Play;FPS;ActionBFree to Play;FPS;MultiplayerB!Free to Play;Fighting;MultiplayerBFree to Play;Fishing;SimulationB#Free to Play;Great Soundtrack;IndieBFree to Play;Horror;DifficultB Free to Play;Horror;SingleplayerB Free to Play;Hunting;MultiplayerBFree to Play;Indie;AtmosphericBFree to Play;Indie;PlatformerB)Free to Play;MMORPG;Massively MultiplayerBFree to Play;MMORPG;RPGBFree to Play;MOBA;MultiplayerB&Free to Play;Massively Multiplayer;FPSB1Free to Play;Massively Multiplayer;Pixel GraphicsB&Free to Play;Massively Multiplayer;RPGB6Free to Play;Massively Multiplayer;Turn-Based StrategyBFree to Play;Mechs;MultiplayerBFree to Play;Memes;ClickerBFree to Play;Multiplayer;ActionBFree to Play;Multiplayer;Co-opBFree to Play;Multiplayer;FPSB!Free to Play;Multiplayer;FightingBFree to Play;Multiplayer;HorrorB.Free to Play;Multiplayer;Massively MultiplayerBFree to Play;Multiplayer;RobotsB Free to Play;Multiplayer;ShooterBFree to Play;Open World;ActionB#Free to Play;Open World;MultiplayerB"Free to Play;Pixel Graphics;ActionB%Free to Play;Pixel Graphics;AdventureB'Free to Play;Psychological Horror;IndieBFree to Play;Puzzle;Co-opB&Free to Play;RPG;Massively MultiplayerB Free to Play;Relaxing;SimulationBFree to Play;Retro;PlatformerBFree to Play;Robots;BuildingBFree to Play;Sci-fi;SpaceB"Free to Play;Sexual Content;NudityB Free to Play;Shooter;MultiplayerBFree to Play;Simulation;ShooterBFree to Play;Space;ActionB&Free to Play;Story Rich;Choices MatterB$Free to Play;Story Rich;SingleplayerB!Free to Play;Strategy;MultiplayerB,Free to Play;Superhero;Massively MultiplayerB#Free to Play;Survival;Base-BuildingB!Free to Play;Survival;MultiplayerBFree to Play;Survival;ZombiesB&Free to Play;Tower Defense;MultiplayerB$Free to Play;Walking Simulator;IndieB%Free to Play;World War II;MultiplayerBFree to Play;World War II;NavalB Free to Play;Zombies;MultiplayerBFree to Play;Zombies;SurvivalBFunny;Multiplayer;Co-opBFunny;Multiplayer;FightingBFunny;Multiplayer;First-PersonBGore;Funny;HorrorBGore;Violent;ActionB"Grand Strategy;Strategy;HistoricalB Grand Strategy;Strategy;MedievalBGreat Soundtrack;2D;PlatformerB&Great Soundtrack;Action;Hack and SlashBGreat Soundtrack;Action;ViolentB!Great Soundtrack;Difficult;ActionB)Great Soundtrack;Female Protagonist;IndieB$Great Soundtrack;Gore;Pixel GraphicsBGreat Soundtrack;Indie;ActionB*Great Soundtrack;Story Rich;Choices MatterB.Great Soundtrack;Story Rich;Female ProtagonistB!Great Soundtrack;Story Rich;IndieB Hack and Slash;Action;HistoricalBHacking;Simulation;IndieBHidden Object;Casual;IndieBHorror;Action;Sci-fiBHorror;Adventure;AtmosphericBHorror;Adventure;NudityBHorror;Atmospheric;First-PersonBHorror;Atmospheric;IndieBHorror;Atmospheric;Story RichBHorror;FPS;ActionBHorror;First-Person;AtmosphericB Horror;First-Person;SingleplayerB#Horror;First-Person;Survival HorrorBHorror;Free to Play;Co-opBHorror;Free to Play;CuteBHorror;Indie;First-PersonBHorror;Indie;Free to PlayB"Horror;Multiplayer;Survival HorrorB'Horror;Psychological Horror;AtmosphericB+Horror;Psychological Horror;Survival HorrorB#Horror;Singleplayer;Survival HorrorB#Horror;Survival Horror;First-PersonB"Horror;Survival Horror;MultiplayerB+Horror;Survival Horror;Psychological HorrorBHorror;Survival Horror;Sci-fiB#Horror;Survival Horror;SingleplayerBHunting;Open World;MultiplayerBIlluminati;Open World;HackingBIndie;Action;CasualBIndie;Action;FPSBIndie;Casual;PuzzleBIndie;Casual;RelaxingBIndie;Dungeon Crawler;RPGB!Indie;Horror;Psychological HorrorBIndie;Platformer;AdventureBIndie;Platformer;Local Co-OpBIndie;Platformer;PuzzleBIndie;Political;SimulationBIndie;RPG;Rogue-likeBIndie;Replay Value;Rogue-likeB!Indie;Story Rich;Great SoundtrackBIndie;Strategy;Choices MatterB+Inventory Management;Survival Horror;ActionBJRPG;Anime;RPGBJRPG;Linear;RPGBJRPG;RPG;ClassicBJRPG;Story Rich;RPGBLEGO;Open World;BuildingBLEGO;Open World;SuperheroB#Local Co-Op;Local Multiplayer;Co-opBLocal Multiplayer;Indie;ActionB)MMORPG;Massively Multiplayer;Free to PlayB MMORPG;Massively Multiplayer;RPGBMOBA;Multiplayer;PlatformerBMagic;Co-op;AdventureBManagement;Pixel Graphics;IndieB"Management;Pixel Graphics;StrategyBManagement;Racing;SimulationB'Massively Multiplayer;MMORPG;Open WorldB Massively Multiplayer;RPG;MMORPGBMedieval;Action;SwordplayBMedieval;Multiplayer;ActionBMedieval;Open World;RPGBMedieval;RPG;Open WorldBMemes;Adventure;SingleplayerBMemes;Cute;CasualB'Metroidvania;Difficult;Great SoundtrackBMetroidvania;Platformer;ActionB$Multiplayer;Action;Local MultiplayerBMultiplayer;Casual;ComedyBMultiplayer;Casual;Local Co-OpBMultiplayer;FPS;ZombiesBMultiplayer;Free to Play;HorrorBMultiplayer;Funny;Battle RoyaleBMultiplayer;Funny;FightingB#Multiplayer;Funny;Local MultiplayerBMultiplayer;Indie;ActionB$Multiplayer;Racing;Local MultiplayerBMultiplayer;Racing;SoccerBMultiplayer;Strategy;FPSBMusic;Education;SimulationBMusic;Indie;Shoot 'Em UpBMusic;Rhythm;IndieBMystery;Detective;AtmosphericBNudity;Anime;Free to PlayBNudity;Mature;AnimeB Nudity;Story Rich;Pixel GraphicsBOffroad;Driving;SimulationBOpen World;Action;1980sBOpen World;Action;AdventureBOpen World;Action;BowlingB)Open World;Action;Character CustomizationBOpen World;Action;ClassicBOpen World;Action;ComedyBOpen World;Action;DestructionBOpen World;Action;FPSBOpen World;Action;Martial ArtsBOpen World;Action;MultiplayerB"Open World;Action;Post-apocalypticBOpen World;Action;RPGBOpen World;Action;SandboxBOpen World;Assassin;ActionB'Open World;Atmospheric;Post-apocalypticBOpen World;Crafting;RPGBOpen World;FPS;ActionBOpen World;Hacking;ActionBOpen World;Medieval;SurvivalBOpen World;Multiplayer;ActionB+Open World;Multiplayer;Third-Person ShooterBOpen World;Parkour;AssassinB'Open World;Post-apocalyptic;ExplorationBOpen World;Post-apocalyptic;RPGBOpen World;RPG;AdventureBOpen World;RPG;FantasyBOpen World;RPG;Post-apocalypticBOpen World;RPG;Story RichBOpen World;Sandbox;RPGBOpen World;Shooter;ActionBOpen World;Space;ExplorationBOpen World;Survival;ActionBOpen World;Survival;MultiplayerBParkour;Action;First-PersonBParkour;First-Person;ActionBParkour;Relaxing;3D PlatformerBPirates;Open World;ActionB+Pixel Graphics;Great Soundtrack;AtmosphericB.Pixel Graphics;Metroidvania;Female ProtagonistBPixel Graphics;Sandbox;CraftingBPixel Graphics;Stealth;StrategyB Pixel Graphics;Strategy;CraftingBPlatformer;Adventure;PuzzleBPlatformer;Comedy;AdventureBPlatformer;Fantasy;PuzzleB"Platformer;Great Soundtrack;ActionBPlatformer;Indie;ActionB!Platformer;Indie;Great SoundtrackBPlatformer;Indie;NarrationBPlatformer;Indie;PuzzleBPlatformer;Indie;SatireBPlatformer;Metroidvania;IndieBPlatformer;Mining;SteampunkB#Platformer;Pixel Graphics;DifficultB*Platformer;Pixel Graphics;Great SoundtrackBPlatformer;Pixel Graphics;IndieBPoint & Click;Adventure;IndieBPoint & Click;Adventure;PuzzleB Post-apocalyptic;Atmospheric;FPSB'Psychological Horror;Anime;Visual NovelB-Psychological Horror;Story Rich;Point & ClickBPuzzle;Adventure;IndieBPuzzle;Casual;AdventureBPuzzle;Casual;IndieBPuzzle;Exploration;First-PersonBPuzzle;First-Person;IndieBPuzzle;First-Person;Sci-fiBPuzzle;Free to Play;AnimeBPuzzle;Indie;CasualB!Puzzle;Indie;Psychological HorrorBPuzzle;Mystery;AdventureBPuzzle;Mystery;Point & ClickBPuzzle;Platformer;IndieBPuzzle;Sci-fi;AtmosphericBPvP;Multiplayer;CompetitiveBQuick-Time Events;Rome;ActionBRPG;Action RPG;Hack and SlashBRPG;Action RPG;Story RichBRPG;Action;First-PersonBRPG;Action;Third PersonBRPG;Adventure;ActionBRPG;Classic;AdventureBRPG;Comedy;AdventureBRPG;Comedy;Dark HumorBRPG;Cyberpunk;HorrorBRPG;Cyberpunk;Turn-BasedBRPG;Dark Fantasy;DifficultBRPG;Fantasy;ClassicBRPG;Fantasy;Great SoundtrackBRPG;Fantasy;IsometricBRPG;Fantasy;MatureBRPG;Fantasy;Open WorldBRPG;Fantasy;SingleplayerBRPG;Fantasy;Story RichBRPG;Hack and Slash;Action RPGBRPG;Indie;AdventureBRPG;Indie;PlatformerBRPG;Open World;ActionB&RPG;Open World;Character CustomizationBRPG;Open World;FantasyBRPG;Open World;JRPGBRPG;Open World;MMORPGBRPG;Open World;SingleplayerBRPG;Post-apocalyptic;ClassicBRPG;Post-apocalyptic;Turn-BasedBRPG;Rogue-like;IndieBRPG;Sci-fi;Story RichBRPG;Simulation;Pixel GraphicsBRPG;Star Wars;Sci-fiBRPG;Star Wars;Story RichBRPG;Story Rich;Choices MatterBRPG;Turn-Based Combat;AdventureBRPG;Turn-Based;Co-opBRPG;Turn-Based;Story RichBRPG;Turn-Based;StrategyBRPG;Vampire;Cult ClassicBRacing;Action;MultiplayerBRacing;Destruction;MultiplayerBRacing;Destruction;SimulationBRacing;Driving;SimulationBRacing;Free to Play;MultiplayerBRacing;Indie;SingleplayerBRacing;Multiplayer;Open WorldBRacing;Open World;DrivingBRacing;Open World;MultiplayerBRacing;Simulation;DrivingBRacing;Sports;MultiplayerBRacing;Sports;SimulationBRealistic;World War II;FPSB Relaxing;Atmospheric;ExplorationB"Rhythm;Rogue-like;Great SoundtrackBRogue-like;Indie;ActionBRogue-like;Indie;PlatformerBRogue-like;Indie;Replay ValueB&Rogue-like;Pixel Graphics;MetroidvaniaBRogue-like;Platformer;IndieBRogue-like;Space;IndieB!Rogue-like;Strategy;Tower DefenseBSandbox;Adventure;SurvivalBSandbox;Crafting;SurvivalBSandbox;Multiplayer;FunnyBSci-fi;Space;ActionBSexual Content;Nudity;AnimeBSexual Content;Nudity;ComedyBSexual Content;Nudity;MatureBShooter;Free to Play;FPSBSimulation;Action;MilitaryBSimulation;Building;DrivingBSimulation;Building;IndieBSimulation;Building;ManagementBSimulation;Building;RealisticB Simulation;Building;SingleplayerB*Simulation;Character Customization;SandboxBSimulation;Comedy;IndieBSimulation;Driving;Open WorldBSimulation;Flight;Free to PlayBSimulation;Flight;RealisticBSimulation;Funny;ComedyBSimulation;Gore;MultiplayerBSimulation;Indie;CasualBSimulation;Indie;ManagementBSimulation;Indie;RPGBSimulation;Management;BuildingBSimulation;Management;EconomyBSimulation;Management;IndieBSimulation;Memes;FunnyBSimulation;Military;MultiplayerB"Simulation;Multiplayer;AgricultureB!Simulation;Multiplayer;Open WorldBSimulation;Offroad;PhysicsBSimulation;Politics;StrategyBSimulation;Sandbox;BuildingBSimulation;Sandbox;Open WorldB"Simulation;Singleplayer;ManagementBSimulation;Trains;SingleplayerBSniper;Action;FPSBSniper;Action;StealthBSniper;Multiplayer;ShooterB!Souls-like;Metroidvania;DifficultBSouls-like;RPG;ActionBSpace Sim;Space;Open WorldBSpace;Action;Sci-fiB"Space;Massively Multiplayer;Sci-fiBSpace;Sandbox;BuildingBSpace;Sci-fi;Open WorldBSpace;Simulation;Free to PlayBSpace;Simulation;SandboxBSpace;Strategy;Grand StrategyBSpace;Strategy;RTSBSports;Basketball;GamblingBSports;Free to Play;MultiplayerBSports;Open World;MultiplayerBSports;Simulation;ManagementBStar Wars;Action;FPSBStar Wars;Action;MultiplayerBStar Wars;Action;Sci-fiBStealth;Action;AdventureBStealth;Action;AssassinBStealth;Action;Co-opBStealth;Action;First-PersonBStealth;Action;ShortBStealth;Action;Third PersonBStealth;Fantasy;SingleplayerBStealth;First-Person;ActionBStealth;Open World;Story RichBStealth;Puzzle;IndieBStealth;Tactical;NinjaB Steampunk;Team-Based;MultiplayerBStory Rich;Action;AtmosphericB'Story Rich;Atmospheric;Great SoundtrackB(Story Rich;Atmospheric;Walking SimulatorB*Story Rich;Choices Matter;Great SoundtrackB&Story Rich;Choices Matter;SupernaturalB!Story Rich;Great Soundtrack;IndieB*Story Rich;Great Soundtrack;Pixel GraphicsB*Story Rich;Pixel Graphics;Great SoundtrackB&Story Rich;Third-Person Shooter;ActionB"Story Rich;Walking Simulator;IndieBStrategy;4X;SpaceB)Strategy;City Builder;Resource ManagementB Strategy;City Builder;SimulationB$Strategy;Classic;Turn-Based StrategyBStrategy;Fantasy;RTSBStrategy;Fantasy;WarBStrategy;Free to Play;MedievalB Strategy;Funny;Turn-Based CombatB"Strategy;Historical;Grand StrategyBStrategy;Historical;MilitaryBStrategy;Historical;RomeBStrategy;Historical;WarBStrategy;Management;Story RichBStrategy;Mechs;Turn-BasedBStrategy;Medieval;HistoricalBStrategy;Medieval;RTSBStrategy;Military;MultiplayerBStrategy;Multiplayer;MysteryB#Strategy;Multiplayer;Pixel GraphicsB Strategy;Pixel Graphics;SurvivalBStrategy;RPG;Turn-Based CombatBStrategy;RTS;Base-BuildingBStrategy;RTS;ClassicBStrategy;RTS;MilitaryBStrategy;RTS;MultiplayerBStrategy;RTS;MythologyBStrategy;RTS;Sci-fiBStrategy;RTS;SpaceBStrategy;RTS;World War IIB Strategy;Simulation;City BuilderBStrategy;Simulation;IndieBStrategy;Simulation;RTSBStrategy;Space;RTSBStrategy;Space;Sci-fiB"Strategy;Space;Turn-Based StrategyBStrategy;Star Wars;RTSB#Strategy;Turn-Based Combat;MedievalB$Strategy;Turn-Based Strategy;FantasyB'Strategy;Turn-Based Strategy;HistoricalB+Strategy;Turn-Based Strategy;Pixel GraphicsB#Strategy;Turn-Based Strategy;Sci-fiBStrategy;Turn-Based;4XBStrategy;Turn-Based;TacticalBStrategy;Warhammer 40K;RTSB$Strategy;World War II;Grand StrategyBStrategy;World War II;RTSBStrategy;World War II;WarBSurvival Horror;Zombies;HorrorBSurvival;Crafting;MultiplayerB*Survival;Massively Multiplayer;MultiplayerBSurvival;Multiplayer;Co-opBSurvival;Multiplayer;ZombiesBSurvival;Open World;ActionBSurvival;Open World;CraftingBSurvival;Open World;ExplorationBSurvival;Open World;HorrorBSurvival;Open World;MedievalBSurvival;Open World;MultiplayerB&Survival;Post-apocalyptic;SingleplayerBSurvival;Sandbox;Open WorldBSurvival;Shooter;MultiplayerBSurvival;War;AtmosphericBSurvival;Zombies;CraftingBSurvival;Zombies;MultiplayerBSurvival;Zombies;Open WorldBTactical;Strategy;Top-DownBTanks;Free to Play;MultiplayerBTower Defense;Action;StrategyBTower Defense;Co-op;ActionBTower Defense;FPS;Co-opBTower Defense;RPG;Co-opBTower Defense;Strategy;IndieBTower Defense;Zombies;StrategyB%Turn-Based Combat;RPG;Dungeon CrawlerB'Turn-Based Strategy;Strategy;Turn-BasedB%Turn-Based Strategy;Tactical;StrategyB(Utilities;Software;Design & IllustrationBUtilities;Software;SingleplayerBVampire;RPG;Choices MatterBVisual Novel;Anime;Dating SimB%Visual Novel;Anime;Female ProtagonistBVisual Novel;Anime;Story RichB(Visual Novel;Dating Sim;Multiple EndingsB&Visual Novel;Dating Sim;Sexual ContentB$Visual Novel;Free to Play;Story RichBWalking Simulator;Short;IndieB)Warhammer 40K;Action;Third-Person ShooterBWarhammer 40K;Strategy;RTSB)Warhammer 40K;Third-Person Shooter;ActionBWorld War I;FPS;MultiplayerBWorld War II;FPS;MultiplayerBZombies;Action;Co-opBZombies;Action;Open WorldBZombies;Co-op;GoreBZombies;Free to Play;SurvivalBZombies;Horror;Survival HorrorBZombies;Platformer;IndieBZombies;Survival Horror;HorrorBZombies;Survival;ActionBZombies;Survival;MultiplayerBZombies;Survival;Open WorldBZombies;World War II;FPS
?/
Const_26Const*
_output_shapes	
:?*
dtype0	*?.
value?.B?.	?"?.                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?
Const_27Const*
_output_shapes
:
*
dtype0*?
value?B?
B100000-200000B1000000-2000000B10000000-20000000B200000-500000B2000000-5000000B20000000-50000000B50000-100000B500000-1000000B5000000-10000000B50000000-100000000
?
Const_28Const*
_output_shapes
:
*
dtype0	*e
value\BZ	
"P                                                        	       
       
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_11Const_12*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13858
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_13Const_14*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13866
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_2Const_15Const_16*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13874
?
StatefulPartitionedCall_3StatefulPartitionedCallhash_table_3Const_17Const_18*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13882
?
StatefulPartitionedCall_4StatefulPartitionedCallhash_table_4Const_19Const_20*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13890
?
StatefulPartitionedCall_5StatefulPartitionedCallhash_table_5Const_21Const_22*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13898
?
StatefulPartitionedCall_6StatefulPartitionedCallhash_table_6Const_23Const_24*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13906
?
StatefulPartitionedCall_7StatefulPartitionedCallhash_table_7Const_25Const_26*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13914
?
StatefulPartitionedCall_8StatefulPartitionedCallhash_table_8Const_27Const_28*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_13922
?
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6^StatefulPartitionedCall_7^StatefulPartitionedCall_8
?H
Const_29Const"/device:CPU:0*
_output_shapes
: *
dtype0*?H
value?HB?H B?H
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-0
layer-18
layer_with_weights-1
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
layer-0
layer-1
layer-2
layer-3
layer-4

layer-5
layer-6
layer-7
layer-8
	layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
 layer-23
!layer-24
"layer-25
#layer-26
$layer-27
%layer_with_weights-0
%layer-28
&layer-29
'layer-30
(layer-31
)layer-32
*layer-33
+layer-34
,layer-35
-layer-36
.layer-37
/layer-38
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6	variables
7regularization_losses
8trainable_variables
9	keras_api
?
:iter

;beta_1

<beta_2
	=decay
>learning_rateBm?Cm?Dm?Em?Bv?Cv?Dv?Ev?
1
?0
@1
A2
B3
C4
D5
E6
 

B0
C1
D2
E3
?
Fnon_trainable_variables
Glayer_metrics
	variables
regularization_losses
Hlayer_regularization_losses
Imetrics

Jlayers
trainable_variables
 
R
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
!
Olookup_table
P	keras_api
!
Qlookup_table
R	keras_api
!
Slookup_table
T	keras_api
!
Ulookup_table
V	keras_api
!
Wlookup_table
X	keras_api
!
Ylookup_table
Z	keras_api
!
[lookup_table
\	keras_api
!
]lookup_table
^	keras_api
!
_lookup_table
`	keras_api
?
a
_keep_axis
b_reduce_axis
c_reduce_axis_mask
d_broadcast_shape
?mean
?
adapt_mean
@variance
@adapt_variance
	Acount
e	keras_api
R
f	variables
gregularization_losses
htrainable_variables
i	keras_api
R
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
R
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
R
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
R
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
R
z	variables
{regularization_losses
|trainable_variables
}	keras_api
T
~	variables
regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api

?0
@1
A2
 
 
?
?non_trainable_variables
?layer_metrics
0	variables
1regularization_losses
 ?layer_regularization_losses
?metrics
?layers
2trainable_variables
l

Bkernel
Cbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Dkernel
Ebias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

B0
C1
D2
E3
 

B0
C1
D2
E3
?
?non_trainable_variables
?layer_metrics
6	variables
7regularization_losses
 ?layer_regularization_losses
?metrics
?layers
8trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
@>
VARIABLE_VALUEmean&variables/0/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEvariance&variables/1/.ATTRIBUTES/VARIABLE_VALUE
A?
VARIABLE_VALUEcount&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
A2
 
 

?0
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 
 
 
?
?non_trainable_variables
?layer_metrics
K	variables
Lregularization_losses
 ?layer_regularization_losses
?metrics
?layers
Mtrainable_variables

?_initializer
 

?_initializer
 

?_initializer
 

?_initializer
 

?_initializer
 

?_initializer
 

?_initializer
 

?_initializer
 

?_initializer
 
 
 
 
 
 
 
 
 
?
?non_trainable_variables
?layer_metrics
f	variables
gregularization_losses
 ?layer_regularization_losses
?metrics
?layers
htrainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
j	variables
kregularization_losses
 ?layer_regularization_losses
?metrics
?layers
ltrainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
n	variables
oregularization_losses
 ?layer_regularization_losses
?metrics
?layers
ptrainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
r	variables
sregularization_losses
 ?layer_regularization_losses
?metrics
?layers
ttrainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
v	variables
wregularization_losses
 ?layer_regularization_losses
?metrics
?layers
xtrainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
z	variables
{regularization_losses
 ?layer_regularization_losses
?metrics
?layers
|trainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
~	variables
regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
 
 
 
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables

?0
@1
A2
 
 
 
?
0
1
2
3
4

5
6
7
8
	9
10
11
12
13
14
15
16
17
18
19
20
21
22
 23
!24
"25
#26
$27
%28
&29
'30
(31
)32
*33
+34
,35
-36
.37
/38

B0
C1
 

B0
C1
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables

D0
E1
 

D0
E1
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
 
 
 
 

40
51
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_achievementsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_appidPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_average_playtimePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_categoriesPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_developerPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_englishPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_genresPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_median_playtimePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_namePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_negative_ratingsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_ownersPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_platformsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
 serving_default_positive_ratingsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
x
serving_default_pricePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_publisherPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_release_datePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_required_agePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_steamspy_tagsPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_9StatefulPartitionedCallserving_default_achievementsserving_default_appid serving_default_average_playtimeserving_default_categoriesserving_default_developerserving_default_englishserving_default_genresserving_default_median_playtimeserving_default_name serving_default_negative_ratingsserving_default_ownersserving_default_platforms serving_default_positive_ratingsserving_default_priceserving_default_publisherserving_default_release_dateserving_default_required_ageserving_default_steamspy_tagshash_table_8Consthash_table_7Const_1hash_table_6Const_2hash_table_5Const_3hash_table_4Const_4hash_table_3Const_5hash_table_2Const_6hash_table_1Const_7
hash_tableConst_8Const_9Const_10dense/kernel
dense/biasdense_1/kerneldense_1/bias*5
Tin.
,2*									*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
&'()*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_11499
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_10StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst_29*#
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_14075
?
StatefulPartitionedCall_11StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratemeanvariancecountdense/kernel
dense/biasdense_1/kerneldense_1/biastotalcount_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_14151??(
?
l
3__inference_category_encoding_1_layer_call_fn_13348

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_1_layer_call_and_return_conditional_losses_98942
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_concatenate_layer_call_fn_13224
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_98152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8
?
:
__inference__creator_13711
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6523*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
l
3__inference_category_encoding_5_layer_call_fn_13504

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_100382
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_138747
3key_value_init6574_lookuptableimportv2_table_handle/
+key_value_init6574_lookuptableimportv2_keys1
-key_value_init6574_lookuptableimportv2_values	
identity??&key_value_init6574/LookupTableImportV2?
&key_value_init6574/LookupTableImportV2LookupTableImportV23key_value_init6574_lookuptableimportv2_table_handle+key_value_init6574_lookuptableimportv2_keys-key_value_init6574_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6574/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6574/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6574/LookupTableImportV2&key_value_init6574/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
*__inference_sequential_layer_call_fn_13184

inputs
unknown:	? @
	unknown_0:@
	unknown_1:@
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_107432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
__inference__initializer_138097
3key_value_init6782_lookuptableimportv2_table_handle/
+key_value_init6782_lookuptableimportv2_keys1
-key_value_init6782_lookuptableimportv2_values	
identity??&key_value_init6782/LookupTableImportV2?
&key_value_init6782/LookupTableImportV2LookupTableImportV23key_value_init6782_lookuptableimportv2_table_handle+key_value_init6782_lookuptableimportv2_keys-key_value_init6782_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6782/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6782/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6782/LookupTableImportV2&key_value_init6782/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
,
__inference__destroyer_13706
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_13211
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????	2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/8
?'
?
B__inference_model_1_layer_call_and_return_conditional_losses_11421
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags
model_11370
model_11372	
model_11374
model_11376	
model_11378
model_11380	
model_11382
model_11384	
model_11386
model_11388	
model_11390
model_11392	
model_11394
model_11396	
model_11398
model_11400	
model_11402
model_11404	
model_11406
model_11408#
sequential_11411:	? @
sequential_11413:@"
sequential_11415:@
sequential_11417:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallachievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tagsmodel_11370model_11372model_11374model_11376model_11378model_11380model_11382model_11384model_11386model_11388model_11390model_11392model_11394model_11396model_11398model_11400model_11402model_11404model_11406model_11408*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_104482
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_11411sequential_11413sequential_11415sequential_11417*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_108032$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?!
?
#__inference_signature_wrapper_11499
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18

unknown_19:	? @

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallachievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*5
Tin.
,2*									*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
&'()*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_97242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?&
?
B__inference_model_1_layer_call_and_return_conditional_losses_10947

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
model_10896
model_10898	
model_10900
model_10902	
model_10904
model_10906	
model_10908
model_10910	
model_10912
model_10914	
model_10916
model_10918	
model_10920
model_10922	
model_10924
model_10926	
model_10928
model_10930	
model_10932
model_10934#
sequential_10937:	? @
sequential_10939:@"
sequential_10941:@
sequential_10943:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17model_10896model_10898model_10900model_10902model_10904model_10906model_10908model_10910model_10912model_10914model_10916model_10918model_10920model_10922model_10924model_10926model_10928model_10930model_10932model_10934*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_101662
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_10937sequential_10939sequential_10941sequential_10943*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_107432$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
j
1__inference_category_encoding_layer_call_fn_13309

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_category_encoding_layer_call_and_return_conditional_losses_98582
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
}
N__inference_category_encoding_2_layer_call_and_return_conditional_losses_13382

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_138987
3key_value_init6730_lookuptableimportv2_table_handle/
+key_value_init6730_lookuptableimportv2_keys1
-key_value_init6730_lookuptableimportv2_values	
identity??&key_value_init6730/LookupTableImportV2?
&key_value_init6730/LookupTableImportV2LookupTableImportV23key_value_init6730_lookuptableimportv2_table_handle+key_value_init6730_lookuptableimportv2_keys-key_value_init6730_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6730/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6730/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6730/LookupTableImportV2&key_value_init6730/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
,
__inference__destroyer_13850
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
l
3__inference_category_encoding_3_layer_call_fn_13426

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_3_layer_call_and_return_conditional_losses_99662
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
:
__inference__creator_13819
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6835*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
? 
|
M__inference_category_encoding_2_layer_call_and_return_conditional_losses_9930

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_10827
dense_input
unknown:	? @
	unknown_0:@
	unknown_1:@
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_108032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????? 
%
_user_specified_namedense_input
??
?
@__inference_model_layer_call_and_return_conditional_losses_10703
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags>
:string_lookup_8_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_4_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_4_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
identity??)category_encoding/StatefulPartitionedCall?+category_encoding_1/StatefulPartitionedCall?+category_encoding_2/StatefulPartitionedCall?+category_encoding_3/StatefulPartitionedCall?+category_encoding_4/StatefulPartitionedCall?+category_encoding_5/StatefulPartitionedCall?+category_encoding_6/StatefulPartitionedCall?+category_encoding_7/StatefulPartitionedCall?+category_encoding_8/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?-string_lookup_1/None_Lookup/LookupTableFindV2?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2?-string_lookup_4/None_Lookup/LookupTableFindV2?-string_lookup_5/None_Lookup/LookupTableFindV2?-string_lookup_6/None_Lookup/LookupTableFindV2?-string_lookup_7/None_Lookup/LookupTableFindV2?-string_lookup_8/None_Lookup/LookupTableFindV2?
-string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_8_none_lookup_lookuptablefindv2_table_handleowners;string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_8/None_Lookup/LookupTableFindV2?
string_lookup_8/IdentityIdentity6string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_8/Identity?
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handlesteamspy_tags;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_7/None_Lookup/LookupTableFindV2?
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_7/Identity?
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handlegenres;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_6/None_Lookup/LookupTableFindV2?
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_6/Identity?
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handle
categories;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_5/None_Lookup/LookupTableFindV2?
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_5/Identity?
-string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_4_none_lookup_lookuptablefindv2_table_handle	platforms;string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_4/None_Lookup/LookupTableFindV2?
string_lookup_4/IdentityIdentity6string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_4/Identity?
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handle	publisher;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_3/None_Lookup/LookupTableFindV2?
string_lookup_3/IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_3/Identity?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handle	developer;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_2/Identity?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handlerelease_date;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlename9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
concatenate/PartitionedCallPartitionedCallappidenglishrequired_ageachievementspositive_ratingsnegative_ratingsaverage_playtimemedian_playtimeprice*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_98152
concatenate/PartitionedCall?
normalization/subSub$concatenate/PartitionedCall:output:0normalization_sub_y*
T0*'
_output_shapes
:?????????	2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:	2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
normalization/truediv?
)category_encoding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_category_encoding_layer_call_and_return_conditional_losses_98582+
)category_encoding/StatefulPartitionedCall?
+category_encoding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0*^category_encoding/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_1_layer_call_and_return_conditional_losses_98942-
+category_encoding_1/StatefulPartitionedCall?
+category_encoding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0,^category_encoding_1/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_2_layer_call_and_return_conditional_losses_99302-
+category_encoding_2/StatefulPartitionedCall?
+category_encoding_3/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_3/Identity:output:0,^category_encoding_2/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_3_layer_call_and_return_conditional_losses_99662-
+category_encoding_3/StatefulPartitionedCall?
+category_encoding_4/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_4/Identity:output:0,^category_encoding_3/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_100022-
+category_encoding_4/StatefulPartitionedCall?
+category_encoding_5/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_5/Identity:output:0,^category_encoding_4/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_100382-
+category_encoding_5/StatefulPartitionedCall?
+category_encoding_6/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_6/Identity:output:0,^category_encoding_5/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_100742-
+category_encoding_6/StatefulPartitionedCall?
+category_encoding_7/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_7/Identity:output:0,^category_encoding_6/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_101102-
+category_encoding_7/StatefulPartitionedCall?
+category_encoding_8/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_8/Identity:output:0,^category_encoding_7/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_101462-
+category_encoding_8/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:02category_encoding/StatefulPartitionedCall:output:04category_encoding_1/StatefulPartitionedCall:output:04category_encoding_2/StatefulPartitionedCall:output:04category_encoding_3/StatefulPartitionedCall:output:04category_encoding_4/StatefulPartitionedCall:output:04category_encoding_5/StatefulPartitionedCall:output:04category_encoding_6/StatefulPartitionedCall:output:04category_encoding_7/StatefulPartitionedCall:output:04category_encoding_8/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_101632
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identity?
NoOpNoOp*^category_encoding/StatefulPartitionedCall,^category_encoding_1/StatefulPartitionedCall,^category_encoding_2/StatefulPartitionedCall,^category_encoding_3/StatefulPartitionedCall,^category_encoding_4/StatefulPartitionedCall,^category_encoding_5/StatefulPartitionedCall,^category_encoding_6/StatefulPartitionedCall,^category_encoding_7/StatefulPartitionedCall,^category_encoding_8/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2.^string_lookup_1/None_Lookup/LookupTableFindV2.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2.^string_lookup_4/None_Lookup/LookupTableFindV2.^string_lookup_5/None_Lookup/LookupTableFindV2.^string_lookup_6/None_Lookup/LookupTableFindV2.^string_lookup_7/None_Lookup/LookupTableFindV2.^string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	2V
)category_encoding/StatefulPartitionedCall)category_encoding/StatefulPartitionedCall2Z
+category_encoding_1/StatefulPartitionedCall+category_encoding_1/StatefulPartitionedCall2Z
+category_encoding_2/StatefulPartitionedCall+category_encoding_2/StatefulPartitionedCall2Z
+category_encoding_3/StatefulPartitionedCall+category_encoding_3/StatefulPartitionedCall2Z
+category_encoding_4/StatefulPartitionedCall+category_encoding_4/StatefulPartitionedCall2Z
+category_encoding_5/StatefulPartitionedCall+category_encoding_5/StatefulPartitionedCall2Z
+category_encoding_6/StatefulPartitionedCall+category_encoding_6/StatefulPartitionedCall2Z
+category_encoding_7/StatefulPartitionedCall+category_encoding_7/StatefulPartitionedCall2Z
+category_encoding_8/StatefulPartitionedCall+category_encoding_8/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV22^
-string_lookup_4/None_Lookup/LookupTableFindV2-string_lookup_4/None_Lookup/LookupTableFindV22^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV22^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV22^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV22^
-string_lookup_8/None_Lookup/LookupTableFindV2-string_lookup_8/None_Lookup/LookupTableFindV2:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
,
__inference__destroyer_13796
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
,
__inference__destroyer_13724
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_10736

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_10720

inputs1
matmul_readvariableop_resource:	? @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?$
?
'__inference_model_1_layer_call_fn_12339
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18

unknown_19:	? @

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_achievementsinputs_appidinputs_average_playtimeinputs_categoriesinputs_developerinputs_englishinputs_genresinputs_median_playtimeinputs_nameinputs_negative_ratingsinputs_ownersinputs_platformsinputs_positive_ratingsinputs_priceinputs_publisherinputs_release_dateinputs_required_ageinputs_steamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*5
Tin.
,2*									*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
&'()*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_111582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
l
3__inference_category_encoding_6_layer_call_fn_13543

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_100742
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_concatenate_1_layer_call_fn_13650
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_101632
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:??????????:??????????:??????????:??????????:?????????:??????????:??????????:??????????:?????????:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9
?
:
__inference__creator_13783
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6731*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_11849
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tagsD
@model_string_lookup_8_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_8_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_7_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_7_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_6_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_6_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_5_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_5_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_4_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_4_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_3_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_3_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_2_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_2_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_1_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_1_none_lookup_lookuptablefindv2_default_value	B
>model_string_lookup_none_lookup_lookuptablefindv2_table_handleC
?model_string_lookup_none_lookup_lookuptablefindv2_default_value	
model_normalization_sub_y
model_normalization_sqrt_xB
/sequential_dense_matmul_readvariableop_resource:	? @>
0sequential_dense_biasadd_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@
2sequential_dense_1_biasadd_readvariableop_resource:
identity??%model/category_encoding/Assert/Assert?'model/category_encoding_1/Assert/Assert?'model/category_encoding_2/Assert/Assert?'model/category_encoding_3/Assert/Assert?'model/category_encoding_4/Assert/Assert?'model/category_encoding_5/Assert/Assert?'model/category_encoding_6/Assert/Assert?'model/category_encoding_7/Assert/Assert?'model/category_encoding_8/Assert/Assert?1model/string_lookup/None_Lookup/LookupTableFindV2?3model/string_lookup_1/None_Lookup/LookupTableFindV2?3model/string_lookup_2/None_Lookup/LookupTableFindV2?3model/string_lookup_3/None_Lookup/LookupTableFindV2?3model/string_lookup_4/None_Lookup/LookupTableFindV2?3model/string_lookup_5/None_Lookup/LookupTableFindV2?3model/string_lookup_6/None_Lookup/LookupTableFindV2?3model/string_lookup_7/None_Lookup/LookupTableFindV2?3model/string_lookup_8/None_Lookup/LookupTableFindV2?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
3model/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_8_none_lookup_lookuptablefindv2_table_handleinputs_ownersAmodel_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_8/None_Lookup/LookupTableFindV2?
model/string_lookup_8/IdentityIdentity<model/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_8/Identity?
3model/string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_7_none_lookup_lookuptablefindv2_table_handleinputs_steamspy_tagsAmodel_string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_7/None_Lookup/LookupTableFindV2?
model/string_lookup_7/IdentityIdentity<model/string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_7/Identity?
3model/string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_6_none_lookup_lookuptablefindv2_table_handleinputs_genresAmodel_string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_6/None_Lookup/LookupTableFindV2?
model/string_lookup_6/IdentityIdentity<model/string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_6/Identity?
3model/string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_5_none_lookup_lookuptablefindv2_table_handleinputs_categoriesAmodel_string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_5/None_Lookup/LookupTableFindV2?
model/string_lookup_5/IdentityIdentity<model/string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_5/Identity?
3model/string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_4_none_lookup_lookuptablefindv2_table_handleinputs_platformsAmodel_string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_4/None_Lookup/LookupTableFindV2?
model/string_lookup_4/IdentityIdentity<model/string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_4/Identity?
3model/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_3_none_lookup_lookuptablefindv2_table_handleinputs_publisherAmodel_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_3/None_Lookup/LookupTableFindV2?
model/string_lookup_3/IdentityIdentity<model/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_3/Identity?
3model/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_developerAmodel_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_2/None_Lookup/LookupTableFindV2?
model/string_lookup_2/IdentityIdentity<model/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_2/Identity?
3model/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_release_dateAmodel_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_1/None_Lookup/LookupTableFindV2?
model/string_lookup_1/IdentityIdentity<model/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_1/Identity?
1model/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2>model_string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_name?model_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1model/string_lookup/None_Lookup/LookupTableFindV2?
model/string_lookup/IdentityIdentity:model/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
model/string_lookup/Identity?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2inputs_appidinputs_englishinputs_required_ageinputs_achievementsinputs_positive_ratingsinputs_negative_ratingsinputs_average_playtimeinputs_median_playtimeinputs_price&model/concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????	2
model/concatenate/concat?
model/normalization/subSub!model/concatenate/concat:output:0model_normalization_sub_y*
T0*'
_output_shapes
:?????????	2
model/normalization/sub?
model/normalization/SqrtSqrtmodel_normalization_sqrt_x*
T0*
_output_shapes

:	2
model/normalization/Sqrt?
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
model/normalization/Maximum/y?
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
model/normalization/Maximum?
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
model/normalization/truediv?
model/category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model/category_encoding/Const?
model/category_encoding/MaxMax%model/string_lookup/Identity:output:0&model/category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Max?
model/category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding/Const_1?
model/category_encoding/MinMin%model/string_lookup/Identity:output:0(model/category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Min?
model/category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2 
model/category_encoding/Cast/x?
model/category_encoding/CastCast'model/category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
model/category_encoding/Cast?
model/category_encoding/GreaterGreater model/category_encoding/Cast:y:0$model/category_encoding/Max:output:0*
T0	*
_output_shapes
: 2!
model/category_encoding/Greater?
 model/category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model/category_encoding/Cast_1/x?
model/category_encoding/Cast_1Cast)model/category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding/Cast_1?
$model/category_encoding/GreaterEqualGreaterEqual$model/category_encoding/Min:output:0"model/category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/GreaterEqual?
"model/category_encoding/LogicalAnd
LogicalAnd#model/category_encoding/Greater:z:0(model/category_encoding/GreaterEqual:z:0*
_output_shapes
: 2$
"model/category_encoding/LogicalAnd?
$model/category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502&
$model/category_encoding/Assert/Const?
,model/category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502.
,model/category_encoding/Assert/Assert/data_0?
%model/category_encoding/Assert/AssertAssert&model/category_encoding/LogicalAnd:z:05model/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2'
%model/category_encoding/Assert/Assert?
&model/category_encoding/bincount/ShapeShape%model/string_lookup/Identity:output:0&^model/category_encoding/Assert/Assert*
T0	*
_output_shapes
:2(
&model/category_encoding/bincount/Shape?
&model/category_encoding/bincount/ConstConst&^model/category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2(
&model/category_encoding/bincount/Const?
%model/category_encoding/bincount/ProdProd/model/category_encoding/bincount/Shape:output:0/model/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2'
%model/category_encoding/bincount/Prod?
*model/category_encoding/bincount/Greater/yConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2,
*model/category_encoding/bincount/Greater/y?
(model/category_encoding/bincount/GreaterGreater.model/category_encoding/bincount/Prod:output:03model/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2*
(model/category_encoding/bincount/Greater?
%model/category_encoding/bincount/CastCast,model/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2'
%model/category_encoding/bincount/Cast?
(model/category_encoding/bincount/Const_1Const&^model/category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2*
(model/category_encoding/bincount/Const_1?
$model/category_encoding/bincount/MaxMax%model/string_lookup/Identity:output:01model/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/Max?
&model/category_encoding/bincount/add/yConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&model/category_encoding/bincount/add/y?
$model/category_encoding/bincount/addAddV2-model/category_encoding/bincount/Max:output:0/model/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/add?
$model/category_encoding/bincount/mulMul)model/category_encoding/bincount/Cast:y:0(model/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/mul?
*model/category_encoding/bincount/minlengthConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/minlength?
(model/category_encoding/bincount/MaximumMaximum3model/category_encoding/bincount/minlength:output:0(model/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Maximum?
*model/category_encoding/bincount/maxlengthConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/maxlength?
(model/category_encoding/bincount/MinimumMinimum3model/category_encoding/bincount/maxlength:output:0,model/category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Minimum?
(model/category_encoding/bincount/Const_2Const&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2*
(model/category_encoding/bincount/Const_2?
.model/category_encoding/bincount/DenseBincountDenseBincount%model/string_lookup/Identity:output:0,model/category_encoding/bincount/Minimum:z:01model/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(20
.model/category_encoding/bincount/DenseBincount?
model/category_encoding_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_1/Const?
model/category_encoding_1/MaxMax'model/string_lookup_1/Identity:output:0(model/category_encoding_1/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_1/Max?
!model/category_encoding_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_1/Const_1?
model/category_encoding_1/MinMin'model/string_lookup_1/Identity:output:0*model/category_encoding_1/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_1/Min?
 model/category_encoding_1/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_1/Cast/x?
model/category_encoding_1/CastCast)model/category_encoding_1/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_1/Cast?
!model/category_encoding_1/GreaterGreater"model/category_encoding_1/Cast:y:0&model/category_encoding_1/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_1/Greater?
"model/category_encoding_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_1/Cast_1/x?
 model/category_encoding_1/Cast_1Cast+model/category_encoding_1/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_1/Cast_1?
&model/category_encoding_1/GreaterEqualGreaterEqual&model/category_encoding_1/Min:output:0$model/category_encoding_1/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/GreaterEqual?
$model/category_encoding_1/LogicalAnd
LogicalAnd%model/category_encoding_1/Greater:z:0*model/category_encoding_1/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_1/LogicalAnd?
&model/category_encoding_1/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852(
&model/category_encoding_1/Assert/Const?
.model/category_encoding_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=68520
.model/category_encoding_1/Assert/Assert/data_0?
'model/category_encoding_1/Assert/AssertAssert(model/category_encoding_1/LogicalAnd:z:07model/category_encoding_1/Assert/Assert/data_0:output:0&^model/category_encoding/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_1/Assert/Assert?
(model/category_encoding_1/bincount/ShapeShape'model/string_lookup_1/Identity:output:0(^model/category_encoding_1/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_1/bincount/Shape?
(model/category_encoding_1/bincount/ConstConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_1/bincount/Const?
'model/category_encoding_1/bincount/ProdProd1model/category_encoding_1/bincount/Shape:output:01model/category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_1/bincount/Prod?
,model/category_encoding_1/bincount/Greater/yConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_1/bincount/Greater/y?
*model/category_encoding_1/bincount/GreaterGreater0model/category_encoding_1/bincount/Prod:output:05model/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_1/bincount/Greater?
'model/category_encoding_1/bincount/CastCast.model/category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_1/bincount/Cast?
*model/category_encoding_1/bincount/Const_1Const(^model/category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_1/bincount/Const_1?
&model/category_encoding_1/bincount/MaxMax'model/string_lookup_1/Identity:output:03model/category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/bincount/Max?
(model/category_encoding_1/bincount/add/yConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_1/bincount/add/y?
&model/category_encoding_1/bincount/addAddV2/model/category_encoding_1/bincount/Max:output:01model/category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/bincount/add?
&model/category_encoding_1/bincount/mulMul+model/category_encoding_1/bincount/Cast:y:0*model/category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/bincount/mul?
,model/category_encoding_1/bincount/minlengthConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_1/bincount/minlength?
*model/category_encoding_1/bincount/MaximumMaximum5model/category_encoding_1/bincount/minlength:output:0*model/category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_1/bincount/Maximum?
,model/category_encoding_1/bincount/maxlengthConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_1/bincount/maxlength?
*model/category_encoding_1/bincount/MinimumMinimum5model/category_encoding_1/bincount/maxlength:output:0.model/category_encoding_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_1/bincount/Minimum?
*model/category_encoding_1/bincount/Const_2Const(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_1/bincount/Const_2?
0model/category_encoding_1/bincount/DenseBincountDenseBincount'model/string_lookup_1/Identity:output:0.model/category_encoding_1/bincount/Minimum:z:03model/category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_1/bincount/DenseBincount?
model/category_encoding_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_2/Const?
model/category_encoding_2/MaxMax'model/string_lookup_2/Identity:output:0(model/category_encoding_2/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_2/Max?
!model/category_encoding_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_2/Const_1?
model/category_encoding_2/MinMin'model/string_lookup_2/Identity:output:0*model/category_encoding_2/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_2/Min?
 model/category_encoding_2/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_2/Cast/x?
model/category_encoding_2/CastCast)model/category_encoding_2/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_2/Cast?
!model/category_encoding_2/GreaterGreater"model/category_encoding_2/Cast:y:0&model/category_encoding_2/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_2/Greater?
"model/category_encoding_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_2/Cast_1/x?
 model/category_encoding_2/Cast_1Cast+model/category_encoding_2/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_2/Cast_1?
&model/category_encoding_2/GreaterEqualGreaterEqual&model/category_encoding_2/Min:output:0$model/category_encoding_2/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/GreaterEqual?
$model/category_encoding_2/LogicalAnd
LogicalAnd%model/category_encoding_2/Greater:z:0*model/category_encoding_2/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_2/LogicalAnd?
&model/category_encoding_2/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542(
&model/category_encoding_2/Assert/Const?
.model/category_encoding_2/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=65420
.model/category_encoding_2/Assert/Assert/data_0?
'model/category_encoding_2/Assert/AssertAssert(model/category_encoding_2/LogicalAnd:z:07model/category_encoding_2/Assert/Assert/data_0:output:0(^model/category_encoding_1/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_2/Assert/Assert?
(model/category_encoding_2/bincount/ShapeShape'model/string_lookup_2/Identity:output:0(^model/category_encoding_2/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_2/bincount/Shape?
(model/category_encoding_2/bincount/ConstConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_2/bincount/Const?
'model/category_encoding_2/bincount/ProdProd1model/category_encoding_2/bincount/Shape:output:01model/category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_2/bincount/Prod?
,model/category_encoding_2/bincount/Greater/yConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_2/bincount/Greater/y?
*model/category_encoding_2/bincount/GreaterGreater0model/category_encoding_2/bincount/Prod:output:05model/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_2/bincount/Greater?
'model/category_encoding_2/bincount/CastCast.model/category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_2/bincount/Cast?
*model/category_encoding_2/bincount/Const_1Const(^model/category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_2/bincount/Const_1?
&model/category_encoding_2/bincount/MaxMax'model/string_lookup_2/Identity:output:03model/category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/bincount/Max?
(model/category_encoding_2/bincount/add/yConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_2/bincount/add/y?
&model/category_encoding_2/bincount/addAddV2/model/category_encoding_2/bincount/Max:output:01model/category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/bincount/add?
&model/category_encoding_2/bincount/mulMul+model/category_encoding_2/bincount/Cast:y:0*model/category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/bincount/mul?
,model/category_encoding_2/bincount/minlengthConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_2/bincount/minlength?
*model/category_encoding_2/bincount/MaximumMaximum5model/category_encoding_2/bincount/minlength:output:0*model/category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_2/bincount/Maximum?
,model/category_encoding_2/bincount/maxlengthConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_2/bincount/maxlength?
*model/category_encoding_2/bincount/MinimumMinimum5model/category_encoding_2/bincount/maxlength:output:0.model/category_encoding_2/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_2/bincount/Minimum?
*model/category_encoding_2/bincount/Const_2Const(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_2/bincount/Const_2?
0model/category_encoding_2/bincount/DenseBincountDenseBincount'model/string_lookup_2/Identity:output:0.model/category_encoding_2/bincount/Minimum:z:03model/category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_2/bincount/DenseBincount?
model/category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_3/Const?
model/category_encoding_3/MaxMax'model/string_lookup_3/Identity:output:0(model/category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_3/Max?
!model/category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_3/Const_1?
model/category_encoding_3/MinMin'model/string_lookup_3/Identity:output:0*model/category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_3/Min?
 model/category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_3/Cast/x?
model/category_encoding_3/CastCast)model/category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_3/Cast?
!model/category_encoding_3/GreaterGreater"model/category_encoding_3/Cast:y:0&model/category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_3/Greater?
"model/category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_3/Cast_1/x?
 model/category_encoding_3/Cast_1Cast+model/category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_3/Cast_1?
&model/category_encoding_3/GreaterEqualGreaterEqual&model/category_encoding_3/Min:output:0$model/category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/GreaterEqual?
$model/category_encoding_3/LogicalAnd
LogicalAnd%model/category_encoding_3/Greater:z:0*model/category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_3/LogicalAnd?
&model/category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792(
&model/category_encoding_3/Assert/Const?
.model/category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=47920
.model/category_encoding_3/Assert/Assert/data_0?
'model/category_encoding_3/Assert/AssertAssert(model/category_encoding_3/LogicalAnd:z:07model/category_encoding_3/Assert/Assert/data_0:output:0(^model/category_encoding_2/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_3/Assert/Assert?
(model/category_encoding_3/bincount/ShapeShape'model/string_lookup_3/Identity:output:0(^model/category_encoding_3/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_3/bincount/Shape?
(model/category_encoding_3/bincount/ConstConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_3/bincount/Const?
'model/category_encoding_3/bincount/ProdProd1model/category_encoding_3/bincount/Shape:output:01model/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_3/bincount/Prod?
,model/category_encoding_3/bincount/Greater/yConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_3/bincount/Greater/y?
*model/category_encoding_3/bincount/GreaterGreater0model/category_encoding_3/bincount/Prod:output:05model/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_3/bincount/Greater?
'model/category_encoding_3/bincount/CastCast.model/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_3/bincount/Cast?
*model/category_encoding_3/bincount/Const_1Const(^model/category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_3/bincount/Const_1?
&model/category_encoding_3/bincount/MaxMax'model/string_lookup_3/Identity:output:03model/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/bincount/Max?
(model/category_encoding_3/bincount/add/yConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_3/bincount/add/y?
&model/category_encoding_3/bincount/addAddV2/model/category_encoding_3/bincount/Max:output:01model/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/bincount/add?
&model/category_encoding_3/bincount/mulMul+model/category_encoding_3/bincount/Cast:y:0*model/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/bincount/mul?
,model/category_encoding_3/bincount/minlengthConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_3/bincount/minlength?
*model/category_encoding_3/bincount/MaximumMaximum5model/category_encoding_3/bincount/minlength:output:0*model/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_3/bincount/Maximum?
,model/category_encoding_3/bincount/maxlengthConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_3/bincount/maxlength?
*model/category_encoding_3/bincount/MinimumMinimum5model/category_encoding_3/bincount/maxlength:output:0.model/category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_3/bincount/Minimum?
*model/category_encoding_3/bincount/Const_2Const(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_3/bincount/Const_2?
0model/category_encoding_3/bincount/DenseBincountDenseBincount'model/string_lookup_3/Identity:output:0.model/category_encoding_3/bincount/Minimum:z:03model/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_3/bincount/DenseBincount?
model/category_encoding_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_4/Const?
model/category_encoding_4/MaxMax'model/string_lookup_4/Identity:output:0(model/category_encoding_4/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_4/Max?
!model/category_encoding_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_4/Const_1?
model/category_encoding_4/MinMin'model/string_lookup_4/Identity:output:0*model/category_encoding_4/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_4/Min?
 model/category_encoding_4/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/category_encoding_4/Cast/x?
model/category_encoding_4/CastCast)model/category_encoding_4/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_4/Cast?
!model/category_encoding_4/GreaterGreater"model/category_encoding_4/Cast:y:0&model/category_encoding_4/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_4/Greater?
"model/category_encoding_4/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_4/Cast_1/x?
 model/category_encoding_4/Cast_1Cast+model/category_encoding_4/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_4/Cast_1?
&model/category_encoding_4/GreaterEqualGreaterEqual&model/category_encoding_4/Min:output:0$model/category_encoding_4/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/GreaterEqual?
$model/category_encoding_4/LogicalAnd
LogicalAnd%model/category_encoding_4/Greater:z:0*model/category_encoding_4/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_4/LogicalAnd?
&model/category_encoding_4/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52(
&model/category_encoding_4/Assert/Const?
.model/category_encoding_4/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=520
.model/category_encoding_4/Assert/Assert/data_0?
'model/category_encoding_4/Assert/AssertAssert(model/category_encoding_4/LogicalAnd:z:07model/category_encoding_4/Assert/Assert/data_0:output:0(^model/category_encoding_3/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_4/Assert/Assert?
(model/category_encoding_4/bincount/ShapeShape'model/string_lookup_4/Identity:output:0(^model/category_encoding_4/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_4/bincount/Shape?
(model/category_encoding_4/bincount/ConstConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_4/bincount/Const?
'model/category_encoding_4/bincount/ProdProd1model/category_encoding_4/bincount/Shape:output:01model/category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_4/bincount/Prod?
,model/category_encoding_4/bincount/Greater/yConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_4/bincount/Greater/y?
*model/category_encoding_4/bincount/GreaterGreater0model/category_encoding_4/bincount/Prod:output:05model/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_4/bincount/Greater?
'model/category_encoding_4/bincount/CastCast.model/category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_4/bincount/Cast?
*model/category_encoding_4/bincount/Const_1Const(^model/category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_4/bincount/Const_1?
&model/category_encoding_4/bincount/MaxMax'model/string_lookup_4/Identity:output:03model/category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/bincount/Max?
(model/category_encoding_4/bincount/add/yConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_4/bincount/add/y?
&model/category_encoding_4/bincount/addAddV2/model/category_encoding_4/bincount/Max:output:01model/category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/bincount/add?
&model/category_encoding_4/bincount/mulMul+model/category_encoding_4/bincount/Cast:y:0*model/category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/bincount/mul?
,model/category_encoding_4/bincount/minlengthConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_4/bincount/minlength?
*model/category_encoding_4/bincount/MaximumMaximum5model/category_encoding_4/bincount/minlength:output:0*model/category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_4/bincount/Maximum?
,model/category_encoding_4/bincount/maxlengthConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_4/bincount/maxlength?
*model/category_encoding_4/bincount/MinimumMinimum5model/category_encoding_4/bincount/maxlength:output:0.model/category_encoding_4/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_4/bincount/Minimum?
*model/category_encoding_4/bincount/Const_2Const(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_4/bincount/Const_2?
0model/category_encoding_4/bincount/DenseBincountDenseBincount'model/string_lookup_4/Identity:output:0.model/category_encoding_4/bincount/Minimum:z:03model/category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(22
0model/category_encoding_4/bincount/DenseBincount?
model/category_encoding_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_5/Const?
model/category_encoding_5/MaxMax'model/string_lookup_5/Identity:output:0(model/category_encoding_5/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_5/Max?
!model/category_encoding_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_5/Const_1?
model/category_encoding_5/MinMin'model/string_lookup_5/Identity:output:0*model/category_encoding_5/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_5/Min?
 model/category_encoding_5/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_5/Cast/x?
model/category_encoding_5/CastCast)model/category_encoding_5/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_5/Cast?
!model/category_encoding_5/GreaterGreater"model/category_encoding_5/Cast:y:0&model/category_encoding_5/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_5/Greater?
"model/category_encoding_5/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_5/Cast_1/x?
 model/category_encoding_5/Cast_1Cast+model/category_encoding_5/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_5/Cast_1?
&model/category_encoding_5/GreaterEqualGreaterEqual&model/category_encoding_5/Min:output:0$model/category_encoding_5/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/GreaterEqual?
$model/category_encoding_5/LogicalAnd
LogicalAnd%model/category_encoding_5/Greater:z:0*model/category_encoding_5/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_5/LogicalAnd?
&model/category_encoding_5/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952(
&model/category_encoding_5/Assert/Const?
.model/category_encoding_5/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=49520
.model/category_encoding_5/Assert/Assert/data_0?
'model/category_encoding_5/Assert/AssertAssert(model/category_encoding_5/LogicalAnd:z:07model/category_encoding_5/Assert/Assert/data_0:output:0(^model/category_encoding_4/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_5/Assert/Assert?
(model/category_encoding_5/bincount/ShapeShape'model/string_lookup_5/Identity:output:0(^model/category_encoding_5/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_5/bincount/Shape?
(model/category_encoding_5/bincount/ConstConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_5/bincount/Const?
'model/category_encoding_5/bincount/ProdProd1model/category_encoding_5/bincount/Shape:output:01model/category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_5/bincount/Prod?
,model/category_encoding_5/bincount/Greater/yConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_5/bincount/Greater/y?
*model/category_encoding_5/bincount/GreaterGreater0model/category_encoding_5/bincount/Prod:output:05model/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_5/bincount/Greater?
'model/category_encoding_5/bincount/CastCast.model/category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_5/bincount/Cast?
*model/category_encoding_5/bincount/Const_1Const(^model/category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_5/bincount/Const_1?
&model/category_encoding_5/bincount/MaxMax'model/string_lookup_5/Identity:output:03model/category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/bincount/Max?
(model/category_encoding_5/bincount/add/yConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_5/bincount/add/y?
&model/category_encoding_5/bincount/addAddV2/model/category_encoding_5/bincount/Max:output:01model/category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/bincount/add?
&model/category_encoding_5/bincount/mulMul+model/category_encoding_5/bincount/Cast:y:0*model/category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/bincount/mul?
,model/category_encoding_5/bincount/minlengthConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_5/bincount/minlength?
*model/category_encoding_5/bincount/MaximumMaximum5model/category_encoding_5/bincount/minlength:output:0*model/category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_5/bincount/Maximum?
,model/category_encoding_5/bincount/maxlengthConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_5/bincount/maxlength?
*model/category_encoding_5/bincount/MinimumMinimum5model/category_encoding_5/bincount/maxlength:output:0.model/category_encoding_5/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_5/bincount/Minimum?
*model/category_encoding_5/bincount/Const_2Const(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_5/bincount/Const_2?
0model/category_encoding_5/bincount/DenseBincountDenseBincount'model/string_lookup_5/Identity:output:0.model/category_encoding_5/bincount/Minimum:z:03model/category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_5/bincount/DenseBincount?
model/category_encoding_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_6/Const?
model/category_encoding_6/MaxMax'model/string_lookup_6/Identity:output:0(model/category_encoding_6/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_6/Max?
!model/category_encoding_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_6/Const_1?
model/category_encoding_6/MinMin'model/string_lookup_6/Identity:output:0*model/category_encoding_6/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_6/Min?
 model/category_encoding_6/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_6/Cast/x?
model/category_encoding_6/CastCast)model/category_encoding_6/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_6/Cast?
!model/category_encoding_6/GreaterGreater"model/category_encoding_6/Cast:y:0&model/category_encoding_6/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_6/Greater?
"model/category_encoding_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_6/Cast_1/x?
 model/category_encoding_6/Cast_1Cast+model/category_encoding_6/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_6/Cast_1?
&model/category_encoding_6/GreaterEqualGreaterEqual&model/category_encoding_6/Min:output:0$model/category_encoding_6/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/GreaterEqual?
$model/category_encoding_6/LogicalAnd
LogicalAnd%model/category_encoding_6/Greater:z:0*model/category_encoding_6/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_6/LogicalAnd?
&model/category_encoding_6/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172(
&model/category_encoding_6/Assert/Const?
.model/category_encoding_6/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=21720
.model/category_encoding_6/Assert/Assert/data_0?
'model/category_encoding_6/Assert/AssertAssert(model/category_encoding_6/LogicalAnd:z:07model/category_encoding_6/Assert/Assert/data_0:output:0(^model/category_encoding_5/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_6/Assert/Assert?
(model/category_encoding_6/bincount/ShapeShape'model/string_lookup_6/Identity:output:0(^model/category_encoding_6/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_6/bincount/Shape?
(model/category_encoding_6/bincount/ConstConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_6/bincount/Const?
'model/category_encoding_6/bincount/ProdProd1model/category_encoding_6/bincount/Shape:output:01model/category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_6/bincount/Prod?
,model/category_encoding_6/bincount/Greater/yConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_6/bincount/Greater/y?
*model/category_encoding_6/bincount/GreaterGreater0model/category_encoding_6/bincount/Prod:output:05model/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_6/bincount/Greater?
'model/category_encoding_6/bincount/CastCast.model/category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_6/bincount/Cast?
*model/category_encoding_6/bincount/Const_1Const(^model/category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_6/bincount/Const_1?
&model/category_encoding_6/bincount/MaxMax'model/string_lookup_6/Identity:output:03model/category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/bincount/Max?
(model/category_encoding_6/bincount/add/yConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_6/bincount/add/y?
&model/category_encoding_6/bincount/addAddV2/model/category_encoding_6/bincount/Max:output:01model/category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/bincount/add?
&model/category_encoding_6/bincount/mulMul+model/category_encoding_6/bincount/Cast:y:0*model/category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/bincount/mul?
,model/category_encoding_6/bincount/minlengthConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_6/bincount/minlength?
*model/category_encoding_6/bincount/MaximumMaximum5model/category_encoding_6/bincount/minlength:output:0*model/category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_6/bincount/Maximum?
,model/category_encoding_6/bincount/maxlengthConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_6/bincount/maxlength?
*model/category_encoding_6/bincount/MinimumMinimum5model/category_encoding_6/bincount/maxlength:output:0.model/category_encoding_6/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_6/bincount/Minimum?
*model/category_encoding_6/bincount/Const_2Const(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_6/bincount/Const_2?
0model/category_encoding_6/bincount/DenseBincountDenseBincount'model/string_lookup_6/Identity:output:0.model/category_encoding_6/bincount/Minimum:z:03model/category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_6/bincount/DenseBincount?
model/category_encoding_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_7/Const?
model/category_encoding_7/MaxMax'model/string_lookup_7/Identity:output:0(model/category_encoding_7/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_7/Max?
!model/category_encoding_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_7/Const_1?
model/category_encoding_7/MinMin'model/string_lookup_7/Identity:output:0*model/category_encoding_7/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_7/Min?
 model/category_encoding_7/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_7/Cast/x?
model/category_encoding_7/CastCast)model/category_encoding_7/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_7/Cast?
!model/category_encoding_7/GreaterGreater"model/category_encoding_7/Cast:y:0&model/category_encoding_7/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_7/Greater?
"model/category_encoding_7/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_7/Cast_1/x?
 model/category_encoding_7/Cast_1Cast+model/category_encoding_7/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_7/Cast_1?
&model/category_encoding_7/GreaterEqualGreaterEqual&model/category_encoding_7/Min:output:0$model/category_encoding_7/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/GreaterEqual?
$model/category_encoding_7/LogicalAnd
LogicalAnd%model/category_encoding_7/Greater:z:0*model/category_encoding_7/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_7/LogicalAnd?
&model/category_encoding_7/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442(
&model/category_encoding_7/Assert/Const?
.model/category_encoding_7/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=74420
.model/category_encoding_7/Assert/Assert/data_0?
'model/category_encoding_7/Assert/AssertAssert(model/category_encoding_7/LogicalAnd:z:07model/category_encoding_7/Assert/Assert/data_0:output:0(^model/category_encoding_6/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_7/Assert/Assert?
(model/category_encoding_7/bincount/ShapeShape'model/string_lookup_7/Identity:output:0(^model/category_encoding_7/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_7/bincount/Shape?
(model/category_encoding_7/bincount/ConstConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_7/bincount/Const?
'model/category_encoding_7/bincount/ProdProd1model/category_encoding_7/bincount/Shape:output:01model/category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_7/bincount/Prod?
,model/category_encoding_7/bincount/Greater/yConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_7/bincount/Greater/y?
*model/category_encoding_7/bincount/GreaterGreater0model/category_encoding_7/bincount/Prod:output:05model/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_7/bincount/Greater?
'model/category_encoding_7/bincount/CastCast.model/category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_7/bincount/Cast?
*model/category_encoding_7/bincount/Const_1Const(^model/category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_7/bincount/Const_1?
&model/category_encoding_7/bincount/MaxMax'model/string_lookup_7/Identity:output:03model/category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/bincount/Max?
(model/category_encoding_7/bincount/add/yConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_7/bincount/add/y?
&model/category_encoding_7/bincount/addAddV2/model/category_encoding_7/bincount/Max:output:01model/category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/bincount/add?
&model/category_encoding_7/bincount/mulMul+model/category_encoding_7/bincount/Cast:y:0*model/category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/bincount/mul?
,model/category_encoding_7/bincount/minlengthConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_7/bincount/minlength?
*model/category_encoding_7/bincount/MaximumMaximum5model/category_encoding_7/bincount/minlength:output:0*model/category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_7/bincount/Maximum?
,model/category_encoding_7/bincount/maxlengthConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_7/bincount/maxlength?
*model/category_encoding_7/bincount/MinimumMinimum5model/category_encoding_7/bincount/maxlength:output:0.model/category_encoding_7/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_7/bincount/Minimum?
*model/category_encoding_7/bincount/Const_2Const(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_7/bincount/Const_2?
0model/category_encoding_7/bincount/DenseBincountDenseBincount'model/string_lookup_7/Identity:output:0.model/category_encoding_7/bincount/Minimum:z:03model/category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_7/bincount/DenseBincount?
model/category_encoding_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_8/Const?
model/category_encoding_8/MaxMax'model/string_lookup_8/Identity:output:0(model/category_encoding_8/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_8/Max?
!model/category_encoding_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_8/Const_1?
model/category_encoding_8/MinMin'model/string_lookup_8/Identity:output:0*model/category_encoding_8/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_8/Min?
 model/category_encoding_8/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/category_encoding_8/Cast/x?
model/category_encoding_8/CastCast)model/category_encoding_8/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_8/Cast?
!model/category_encoding_8/GreaterGreater"model/category_encoding_8/Cast:y:0&model/category_encoding_8/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_8/Greater?
"model/category_encoding_8/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_8/Cast_1/x?
 model/category_encoding_8/Cast_1Cast+model/category_encoding_8/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_8/Cast_1?
&model/category_encoding_8/GreaterEqualGreaterEqual&model/category_encoding_8/Min:output:0$model/category_encoding_8/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/GreaterEqual?
$model/category_encoding_8/LogicalAnd
LogicalAnd%model/category_encoding_8/Greater:z:0*model/category_encoding_8/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_8/LogicalAnd?
&model/category_encoding_8/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112(
&model/category_encoding_8/Assert/Const?
.model/category_encoding_8/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=1120
.model/category_encoding_8/Assert/Assert/data_0?
'model/category_encoding_8/Assert/AssertAssert(model/category_encoding_8/LogicalAnd:z:07model/category_encoding_8/Assert/Assert/data_0:output:0(^model/category_encoding_7/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_8/Assert/Assert?
(model/category_encoding_8/bincount/ShapeShape'model/string_lookup_8/Identity:output:0(^model/category_encoding_8/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_8/bincount/Shape?
(model/category_encoding_8/bincount/ConstConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_8/bincount/Const?
'model/category_encoding_8/bincount/ProdProd1model/category_encoding_8/bincount/Shape:output:01model/category_encoding_8/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_8/bincount/Prod?
,model/category_encoding_8/bincount/Greater/yConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_8/bincount/Greater/y?
*model/category_encoding_8/bincount/GreaterGreater0model/category_encoding_8/bincount/Prod:output:05model/category_encoding_8/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_8/bincount/Greater?
'model/category_encoding_8/bincount/CastCast.model/category_encoding_8/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_8/bincount/Cast?
*model/category_encoding_8/bincount/Const_1Const(^model/category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_8/bincount/Const_1?
&model/category_encoding_8/bincount/MaxMax'model/string_lookup_8/Identity:output:03model/category_encoding_8/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/bincount/Max?
(model/category_encoding_8/bincount/add/yConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_8/bincount/add/y?
&model/category_encoding_8/bincount/addAddV2/model/category_encoding_8/bincount/Max:output:01model/category_encoding_8/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/bincount/add?
&model/category_encoding_8/bincount/mulMul+model/category_encoding_8/bincount/Cast:y:0*model/category_encoding_8/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/bincount/mul?
,model/category_encoding_8/bincount/minlengthConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_8/bincount/minlength?
*model/category_encoding_8/bincount/MaximumMaximum5model/category_encoding_8/bincount/minlength:output:0*model/category_encoding_8/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_8/bincount/Maximum?
,model/category_encoding_8/bincount/maxlengthConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_8/bincount/maxlength?
*model/category_encoding_8/bincount/MinimumMinimum5model/category_encoding_8/bincount/maxlength:output:0.model/category_encoding_8/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_8/bincount/Minimum?
*model/category_encoding_8/bincount/Const_2Const(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_8/bincount/Const_2?
0model/category_encoding_8/bincount/DenseBincountDenseBincount'model/string_lookup_8/Identity:output:0.model/category_encoding_8/bincount/Minimum:z:03model/category_encoding_8/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(22
0model/category_encoding_8/bincount/DenseBincount?
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_1/concat/axis?
model/concatenate_1/concatConcatV2model/normalization/truediv:z:07model/category_encoding/bincount/DenseBincount:output:09model/category_encoding_1/bincount/DenseBincount:output:09model/category_encoding_2/bincount/DenseBincount:output:09model/category_encoding_3/bincount/DenseBincount:output:09model/category_encoding_4/bincount/DenseBincount:output:09model/category_encoding_5/bincount/DenseBincount:output:09model/category_encoding_6/bincount/DenseBincount:output:09model/category_encoding_7/bincount/DenseBincount:output:09model/category_encoding_8/bincount/DenseBincount:output:0(model/concatenate_1/concat/axis:output:0*
N
*
T0*(
_output_shapes
:?????????? 2
model/concatenate_1/concat?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#model/concatenate_1/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/BiasAdd?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul!sequential/dense/BiasAdd:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd~
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^model/category_encoding/Assert/Assert(^model/category_encoding_1/Assert/Assert(^model/category_encoding_2/Assert/Assert(^model/category_encoding_3/Assert/Assert(^model/category_encoding_4/Assert/Assert(^model/category_encoding_5/Assert/Assert(^model/category_encoding_6/Assert/Assert(^model/category_encoding_7/Assert/Assert(^model/category_encoding_8/Assert/Assert2^model/string_lookup/None_Lookup/LookupTableFindV24^model/string_lookup_1/None_Lookup/LookupTableFindV24^model/string_lookup_2/None_Lookup/LookupTableFindV24^model/string_lookup_3/None_Lookup/LookupTableFindV24^model/string_lookup_4/None_Lookup/LookupTableFindV24^model/string_lookup_5/None_Lookup/LookupTableFindV24^model/string_lookup_6/None_Lookup/LookupTableFindV24^model/string_lookup_7/None_Lookup/LookupTableFindV24^model/string_lookup_8/None_Lookup/LookupTableFindV2(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 2N
%model/category_encoding/Assert/Assert%model/category_encoding/Assert/Assert2R
'model/category_encoding_1/Assert/Assert'model/category_encoding_1/Assert/Assert2R
'model/category_encoding_2/Assert/Assert'model/category_encoding_2/Assert/Assert2R
'model/category_encoding_3/Assert/Assert'model/category_encoding_3/Assert/Assert2R
'model/category_encoding_4/Assert/Assert'model/category_encoding_4/Assert/Assert2R
'model/category_encoding_5/Assert/Assert'model/category_encoding_5/Assert/Assert2R
'model/category_encoding_6/Assert/Assert'model/category_encoding_6/Assert/Assert2R
'model/category_encoding_7/Assert/Assert'model/category_encoding_7/Assert/Assert2R
'model/category_encoding_8/Assert/Assert'model/category_encoding_8/Assert/Assert2f
1model/string_lookup/None_Lookup/LookupTableFindV21model/string_lookup/None_Lookup/LookupTableFindV22j
3model/string_lookup_1/None_Lookup/LookupTableFindV23model/string_lookup_1/None_Lookup/LookupTableFindV22j
3model/string_lookup_2/None_Lookup/LookupTableFindV23model/string_lookup_2/None_Lookup/LookupTableFindV22j
3model/string_lookup_3/None_Lookup/LookupTableFindV23model/string_lookup_3/None_Lookup/LookupTableFindV22j
3model/string_lookup_4/None_Lookup/LookupTableFindV23model/string_lookup_4/None_Lookup/LookupTableFindV22j
3model/string_lookup_5/None_Lookup/LookupTableFindV23model/string_lookup_5/None_Lookup/LookupTableFindV22j
3model/string_lookup_6/None_Lookup/LookupTableFindV23model/string_lookup_6/None_Lookup/LookupTableFindV22j
3model/string_lookup_7/None_Lookup/LookupTableFindV23model/string_lookup_7/None_Lookup/LookupTableFindV22j
3model/string_lookup_8/None_Lookup/LookupTableFindV23model/string_lookup_8/None_Lookup/LookupTableFindV22R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
l
3__inference_category_encoding_4_layer_call_fn_13465

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_100022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_10743

inputs
dense_10721:	? @
dense_10723:@
dense_1_10737:@
dense_1_10739:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10721dense_10723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_107202
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_10737dense_1_10739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_107362!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
? 
}
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_13460

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinR
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mulz
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximumz
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2
bincount/DenseBincountz
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_139227
3key_value_init6886_lookuptableimportv2_table_handle/
+key_value_init6886_lookuptableimportv2_keys1
-key_value_init6886_lookuptableimportv2_values	
identity??&key_value_init6886/LookupTableImportV2?
&key_value_init6886/LookupTableImportV2LookupTableImportV23key_value_init6886_lookuptableimportv2_table_handle+key_value_init6886_lookuptableimportv2_keys-key_value_init6886_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6886/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6886/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :
:
2P
&key_value_init6886/LookupTableImportV2&key_value_init6886/LookupTableImportV2: 

_output_shapes
:
: 

_output_shapes
:

?
?
E__inference_concatenate_layer_call_and_return_conditional_losses_9815

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????	2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_13197

inputs
unknown:	? @
	unknown_0:@
	unknown_1:@
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_108032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
:
__inference__creator_13801
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6783*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_12199
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tagsD
@model_string_lookup_8_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_8_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_7_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_7_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_6_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_6_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_5_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_5_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_4_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_4_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_3_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_3_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_2_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_2_none_lookup_lookuptablefindv2_default_value	D
@model_string_lookup_1_none_lookup_lookuptablefindv2_table_handleE
Amodel_string_lookup_1_none_lookup_lookuptablefindv2_default_value	B
>model_string_lookup_none_lookup_lookuptablefindv2_table_handleC
?model_string_lookup_none_lookup_lookuptablefindv2_default_value	
model_normalization_sub_y
model_normalization_sqrt_xB
/sequential_dense_matmul_readvariableop_resource:	? @>
0sequential_dense_biasadd_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@
2sequential_dense_1_biasadd_readvariableop_resource:
identity??%model/category_encoding/Assert/Assert?'model/category_encoding_1/Assert/Assert?'model/category_encoding_2/Assert/Assert?'model/category_encoding_3/Assert/Assert?'model/category_encoding_4/Assert/Assert?'model/category_encoding_5/Assert/Assert?'model/category_encoding_6/Assert/Assert?'model/category_encoding_7/Assert/Assert?'model/category_encoding_8/Assert/Assert?1model/string_lookup/None_Lookup/LookupTableFindV2?3model/string_lookup_1/None_Lookup/LookupTableFindV2?3model/string_lookup_2/None_Lookup/LookupTableFindV2?3model/string_lookup_3/None_Lookup/LookupTableFindV2?3model/string_lookup_4/None_Lookup/LookupTableFindV2?3model/string_lookup_5/None_Lookup/LookupTableFindV2?3model/string_lookup_6/None_Lookup/LookupTableFindV2?3model/string_lookup_7/None_Lookup/LookupTableFindV2?3model/string_lookup_8/None_Lookup/LookupTableFindV2?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?
3model/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_8_none_lookup_lookuptablefindv2_table_handleinputs_ownersAmodel_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_8/None_Lookup/LookupTableFindV2?
model/string_lookup_8/IdentityIdentity<model/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_8/Identity?
3model/string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_7_none_lookup_lookuptablefindv2_table_handleinputs_steamspy_tagsAmodel_string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_7/None_Lookup/LookupTableFindV2?
model/string_lookup_7/IdentityIdentity<model/string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_7/Identity?
3model/string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_6_none_lookup_lookuptablefindv2_table_handleinputs_genresAmodel_string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_6/None_Lookup/LookupTableFindV2?
model/string_lookup_6/IdentityIdentity<model/string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_6/Identity?
3model/string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_5_none_lookup_lookuptablefindv2_table_handleinputs_categoriesAmodel_string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_5/None_Lookup/LookupTableFindV2?
model/string_lookup_5/IdentityIdentity<model/string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_5/Identity?
3model/string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_4_none_lookup_lookuptablefindv2_table_handleinputs_platformsAmodel_string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_4/None_Lookup/LookupTableFindV2?
model/string_lookup_4/IdentityIdentity<model/string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_4/Identity?
3model/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_3_none_lookup_lookuptablefindv2_table_handleinputs_publisherAmodel_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_3/None_Lookup/LookupTableFindV2?
model/string_lookup_3/IdentityIdentity<model/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_3/Identity?
3model/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_developerAmodel_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_2/None_Lookup/LookupTableFindV2?
model/string_lookup_2/IdentityIdentity<model/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_2/Identity?
3model/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2@model_string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_release_dateAmodel_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????25
3model/string_lookup_1/None_Lookup/LookupTableFindV2?
model/string_lookup_1/IdentityIdentity<model/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2 
model/string_lookup_1/Identity?
1model/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2>model_string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_name?model_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????23
1model/string_lookup/None_Lookup/LookupTableFindV2?
model/string_lookup/IdentityIdentity:model/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
model/string_lookup/Identity?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2inputs_appidinputs_englishinputs_required_ageinputs_achievementsinputs_positive_ratingsinputs_negative_ratingsinputs_average_playtimeinputs_median_playtimeinputs_price&model/concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????	2
model/concatenate/concat?
model/normalization/subSub!model/concatenate/concat:output:0model_normalization_sub_y*
T0*'
_output_shapes
:?????????	2
model/normalization/sub?
model/normalization/SqrtSqrtmodel_normalization_sqrt_x*
T0*
_output_shapes

:	2
model/normalization/Sqrt?
model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
model/normalization/Maximum/y?
model/normalization/MaximumMaximummodel/normalization/Sqrt:y:0&model/normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
model/normalization/Maximum?
model/normalization/truedivRealDivmodel/normalization/sub:z:0model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
model/normalization/truediv?
model/category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model/category_encoding/Const?
model/category_encoding/MaxMax%model/string_lookup/Identity:output:0&model/category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Max?
model/category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding/Const_1?
model/category_encoding/MinMin%model/string_lookup/Identity:output:0(model/category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding/Min?
model/category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2 
model/category_encoding/Cast/x?
model/category_encoding/CastCast'model/category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
model/category_encoding/Cast?
model/category_encoding/GreaterGreater model/category_encoding/Cast:y:0$model/category_encoding/Max:output:0*
T0	*
_output_shapes
: 2!
model/category_encoding/Greater?
 model/category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model/category_encoding/Cast_1/x?
model/category_encoding/Cast_1Cast)model/category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding/Cast_1?
$model/category_encoding/GreaterEqualGreaterEqual$model/category_encoding/Min:output:0"model/category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/GreaterEqual?
"model/category_encoding/LogicalAnd
LogicalAnd#model/category_encoding/Greater:z:0(model/category_encoding/GreaterEqual:z:0*
_output_shapes
: 2$
"model/category_encoding/LogicalAnd?
$model/category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502&
$model/category_encoding/Assert/Const?
,model/category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502.
,model/category_encoding/Assert/Assert/data_0?
%model/category_encoding/Assert/AssertAssert&model/category_encoding/LogicalAnd:z:05model/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2'
%model/category_encoding/Assert/Assert?
&model/category_encoding/bincount/ShapeShape%model/string_lookup/Identity:output:0&^model/category_encoding/Assert/Assert*
T0	*
_output_shapes
:2(
&model/category_encoding/bincount/Shape?
&model/category_encoding/bincount/ConstConst&^model/category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2(
&model/category_encoding/bincount/Const?
%model/category_encoding/bincount/ProdProd/model/category_encoding/bincount/Shape:output:0/model/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2'
%model/category_encoding/bincount/Prod?
*model/category_encoding/bincount/Greater/yConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2,
*model/category_encoding/bincount/Greater/y?
(model/category_encoding/bincount/GreaterGreater.model/category_encoding/bincount/Prod:output:03model/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2*
(model/category_encoding/bincount/Greater?
%model/category_encoding/bincount/CastCast,model/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2'
%model/category_encoding/bincount/Cast?
(model/category_encoding/bincount/Const_1Const&^model/category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2*
(model/category_encoding/bincount/Const_1?
$model/category_encoding/bincount/MaxMax%model/string_lookup/Identity:output:01model/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/Max?
&model/category_encoding/bincount/add/yConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&model/category_encoding/bincount/add/y?
$model/category_encoding/bincount/addAddV2-model/category_encoding/bincount/Max:output:0/model/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/add?
$model/category_encoding/bincount/mulMul)model/category_encoding/bincount/Cast:y:0(model/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2&
$model/category_encoding/bincount/mul?
*model/category_encoding/bincount/minlengthConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/minlength?
(model/category_encoding/bincount/MaximumMaximum3model/category_encoding/bincount/minlength:output:0(model/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Maximum?
*model/category_encoding/bincount/maxlengthConst&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2,
*model/category_encoding/bincount/maxlength?
(model/category_encoding/bincount/MinimumMinimum3model/category_encoding/bincount/maxlength:output:0,model/category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2*
(model/category_encoding/bincount/Minimum?
(model/category_encoding/bincount/Const_2Const&^model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2*
(model/category_encoding/bincount/Const_2?
.model/category_encoding/bincount/DenseBincountDenseBincount%model/string_lookup/Identity:output:0,model/category_encoding/bincount/Minimum:z:01model/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(20
.model/category_encoding/bincount/DenseBincount?
model/category_encoding_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_1/Const?
model/category_encoding_1/MaxMax'model/string_lookup_1/Identity:output:0(model/category_encoding_1/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_1/Max?
!model/category_encoding_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_1/Const_1?
model/category_encoding_1/MinMin'model/string_lookup_1/Identity:output:0*model/category_encoding_1/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_1/Min?
 model/category_encoding_1/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_1/Cast/x?
model/category_encoding_1/CastCast)model/category_encoding_1/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_1/Cast?
!model/category_encoding_1/GreaterGreater"model/category_encoding_1/Cast:y:0&model/category_encoding_1/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_1/Greater?
"model/category_encoding_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_1/Cast_1/x?
 model/category_encoding_1/Cast_1Cast+model/category_encoding_1/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_1/Cast_1?
&model/category_encoding_1/GreaterEqualGreaterEqual&model/category_encoding_1/Min:output:0$model/category_encoding_1/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/GreaterEqual?
$model/category_encoding_1/LogicalAnd
LogicalAnd%model/category_encoding_1/Greater:z:0*model/category_encoding_1/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_1/LogicalAnd?
&model/category_encoding_1/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852(
&model/category_encoding_1/Assert/Const?
.model/category_encoding_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=68520
.model/category_encoding_1/Assert/Assert/data_0?
'model/category_encoding_1/Assert/AssertAssert(model/category_encoding_1/LogicalAnd:z:07model/category_encoding_1/Assert/Assert/data_0:output:0&^model/category_encoding/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_1/Assert/Assert?
(model/category_encoding_1/bincount/ShapeShape'model/string_lookup_1/Identity:output:0(^model/category_encoding_1/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_1/bincount/Shape?
(model/category_encoding_1/bincount/ConstConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_1/bincount/Const?
'model/category_encoding_1/bincount/ProdProd1model/category_encoding_1/bincount/Shape:output:01model/category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_1/bincount/Prod?
,model/category_encoding_1/bincount/Greater/yConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_1/bincount/Greater/y?
*model/category_encoding_1/bincount/GreaterGreater0model/category_encoding_1/bincount/Prod:output:05model/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_1/bincount/Greater?
'model/category_encoding_1/bincount/CastCast.model/category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_1/bincount/Cast?
*model/category_encoding_1/bincount/Const_1Const(^model/category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_1/bincount/Const_1?
&model/category_encoding_1/bincount/MaxMax'model/string_lookup_1/Identity:output:03model/category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/bincount/Max?
(model/category_encoding_1/bincount/add/yConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_1/bincount/add/y?
&model/category_encoding_1/bincount/addAddV2/model/category_encoding_1/bincount/Max:output:01model/category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/bincount/add?
&model/category_encoding_1/bincount/mulMul+model/category_encoding_1/bincount/Cast:y:0*model/category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_1/bincount/mul?
,model/category_encoding_1/bincount/minlengthConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_1/bincount/minlength?
*model/category_encoding_1/bincount/MaximumMaximum5model/category_encoding_1/bincount/minlength:output:0*model/category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_1/bincount/Maximum?
,model/category_encoding_1/bincount/maxlengthConst(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_1/bincount/maxlength?
*model/category_encoding_1/bincount/MinimumMinimum5model/category_encoding_1/bincount/maxlength:output:0.model/category_encoding_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_1/bincount/Minimum?
*model/category_encoding_1/bincount/Const_2Const(^model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_1/bincount/Const_2?
0model/category_encoding_1/bincount/DenseBincountDenseBincount'model/string_lookup_1/Identity:output:0.model/category_encoding_1/bincount/Minimum:z:03model/category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_1/bincount/DenseBincount?
model/category_encoding_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_2/Const?
model/category_encoding_2/MaxMax'model/string_lookup_2/Identity:output:0(model/category_encoding_2/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_2/Max?
!model/category_encoding_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_2/Const_1?
model/category_encoding_2/MinMin'model/string_lookup_2/Identity:output:0*model/category_encoding_2/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_2/Min?
 model/category_encoding_2/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_2/Cast/x?
model/category_encoding_2/CastCast)model/category_encoding_2/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_2/Cast?
!model/category_encoding_2/GreaterGreater"model/category_encoding_2/Cast:y:0&model/category_encoding_2/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_2/Greater?
"model/category_encoding_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_2/Cast_1/x?
 model/category_encoding_2/Cast_1Cast+model/category_encoding_2/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_2/Cast_1?
&model/category_encoding_2/GreaterEqualGreaterEqual&model/category_encoding_2/Min:output:0$model/category_encoding_2/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/GreaterEqual?
$model/category_encoding_2/LogicalAnd
LogicalAnd%model/category_encoding_2/Greater:z:0*model/category_encoding_2/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_2/LogicalAnd?
&model/category_encoding_2/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542(
&model/category_encoding_2/Assert/Const?
.model/category_encoding_2/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=65420
.model/category_encoding_2/Assert/Assert/data_0?
'model/category_encoding_2/Assert/AssertAssert(model/category_encoding_2/LogicalAnd:z:07model/category_encoding_2/Assert/Assert/data_0:output:0(^model/category_encoding_1/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_2/Assert/Assert?
(model/category_encoding_2/bincount/ShapeShape'model/string_lookup_2/Identity:output:0(^model/category_encoding_2/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_2/bincount/Shape?
(model/category_encoding_2/bincount/ConstConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_2/bincount/Const?
'model/category_encoding_2/bincount/ProdProd1model/category_encoding_2/bincount/Shape:output:01model/category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_2/bincount/Prod?
,model/category_encoding_2/bincount/Greater/yConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_2/bincount/Greater/y?
*model/category_encoding_2/bincount/GreaterGreater0model/category_encoding_2/bincount/Prod:output:05model/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_2/bincount/Greater?
'model/category_encoding_2/bincount/CastCast.model/category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_2/bincount/Cast?
*model/category_encoding_2/bincount/Const_1Const(^model/category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_2/bincount/Const_1?
&model/category_encoding_2/bincount/MaxMax'model/string_lookup_2/Identity:output:03model/category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/bincount/Max?
(model/category_encoding_2/bincount/add/yConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_2/bincount/add/y?
&model/category_encoding_2/bincount/addAddV2/model/category_encoding_2/bincount/Max:output:01model/category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/bincount/add?
&model/category_encoding_2/bincount/mulMul+model/category_encoding_2/bincount/Cast:y:0*model/category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_2/bincount/mul?
,model/category_encoding_2/bincount/minlengthConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_2/bincount/minlength?
*model/category_encoding_2/bincount/MaximumMaximum5model/category_encoding_2/bincount/minlength:output:0*model/category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_2/bincount/Maximum?
,model/category_encoding_2/bincount/maxlengthConst(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_2/bincount/maxlength?
*model/category_encoding_2/bincount/MinimumMinimum5model/category_encoding_2/bincount/maxlength:output:0.model/category_encoding_2/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_2/bincount/Minimum?
*model/category_encoding_2/bincount/Const_2Const(^model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_2/bincount/Const_2?
0model/category_encoding_2/bincount/DenseBincountDenseBincount'model/string_lookup_2/Identity:output:0.model/category_encoding_2/bincount/Minimum:z:03model/category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_2/bincount/DenseBincount?
model/category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_3/Const?
model/category_encoding_3/MaxMax'model/string_lookup_3/Identity:output:0(model/category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_3/Max?
!model/category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_3/Const_1?
model/category_encoding_3/MinMin'model/string_lookup_3/Identity:output:0*model/category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_3/Min?
 model/category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_3/Cast/x?
model/category_encoding_3/CastCast)model/category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_3/Cast?
!model/category_encoding_3/GreaterGreater"model/category_encoding_3/Cast:y:0&model/category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_3/Greater?
"model/category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_3/Cast_1/x?
 model/category_encoding_3/Cast_1Cast+model/category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_3/Cast_1?
&model/category_encoding_3/GreaterEqualGreaterEqual&model/category_encoding_3/Min:output:0$model/category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/GreaterEqual?
$model/category_encoding_3/LogicalAnd
LogicalAnd%model/category_encoding_3/Greater:z:0*model/category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_3/LogicalAnd?
&model/category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792(
&model/category_encoding_3/Assert/Const?
.model/category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=47920
.model/category_encoding_3/Assert/Assert/data_0?
'model/category_encoding_3/Assert/AssertAssert(model/category_encoding_3/LogicalAnd:z:07model/category_encoding_3/Assert/Assert/data_0:output:0(^model/category_encoding_2/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_3/Assert/Assert?
(model/category_encoding_3/bincount/ShapeShape'model/string_lookup_3/Identity:output:0(^model/category_encoding_3/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_3/bincount/Shape?
(model/category_encoding_3/bincount/ConstConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_3/bincount/Const?
'model/category_encoding_3/bincount/ProdProd1model/category_encoding_3/bincount/Shape:output:01model/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_3/bincount/Prod?
,model/category_encoding_3/bincount/Greater/yConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_3/bincount/Greater/y?
*model/category_encoding_3/bincount/GreaterGreater0model/category_encoding_3/bincount/Prod:output:05model/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_3/bincount/Greater?
'model/category_encoding_3/bincount/CastCast.model/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_3/bincount/Cast?
*model/category_encoding_3/bincount/Const_1Const(^model/category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_3/bincount/Const_1?
&model/category_encoding_3/bincount/MaxMax'model/string_lookup_3/Identity:output:03model/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/bincount/Max?
(model/category_encoding_3/bincount/add/yConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_3/bincount/add/y?
&model/category_encoding_3/bincount/addAddV2/model/category_encoding_3/bincount/Max:output:01model/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/bincount/add?
&model/category_encoding_3/bincount/mulMul+model/category_encoding_3/bincount/Cast:y:0*model/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_3/bincount/mul?
,model/category_encoding_3/bincount/minlengthConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_3/bincount/minlength?
*model/category_encoding_3/bincount/MaximumMaximum5model/category_encoding_3/bincount/minlength:output:0*model/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_3/bincount/Maximum?
,model/category_encoding_3/bincount/maxlengthConst(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_3/bincount/maxlength?
*model/category_encoding_3/bincount/MinimumMinimum5model/category_encoding_3/bincount/maxlength:output:0.model/category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_3/bincount/Minimum?
*model/category_encoding_3/bincount/Const_2Const(^model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_3/bincount/Const_2?
0model/category_encoding_3/bincount/DenseBincountDenseBincount'model/string_lookup_3/Identity:output:0.model/category_encoding_3/bincount/Minimum:z:03model/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_3/bincount/DenseBincount?
model/category_encoding_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_4/Const?
model/category_encoding_4/MaxMax'model/string_lookup_4/Identity:output:0(model/category_encoding_4/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_4/Max?
!model/category_encoding_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_4/Const_1?
model/category_encoding_4/MinMin'model/string_lookup_4/Identity:output:0*model/category_encoding_4/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_4/Min?
 model/category_encoding_4/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/category_encoding_4/Cast/x?
model/category_encoding_4/CastCast)model/category_encoding_4/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_4/Cast?
!model/category_encoding_4/GreaterGreater"model/category_encoding_4/Cast:y:0&model/category_encoding_4/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_4/Greater?
"model/category_encoding_4/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_4/Cast_1/x?
 model/category_encoding_4/Cast_1Cast+model/category_encoding_4/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_4/Cast_1?
&model/category_encoding_4/GreaterEqualGreaterEqual&model/category_encoding_4/Min:output:0$model/category_encoding_4/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/GreaterEqual?
$model/category_encoding_4/LogicalAnd
LogicalAnd%model/category_encoding_4/Greater:z:0*model/category_encoding_4/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_4/LogicalAnd?
&model/category_encoding_4/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52(
&model/category_encoding_4/Assert/Const?
.model/category_encoding_4/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=520
.model/category_encoding_4/Assert/Assert/data_0?
'model/category_encoding_4/Assert/AssertAssert(model/category_encoding_4/LogicalAnd:z:07model/category_encoding_4/Assert/Assert/data_0:output:0(^model/category_encoding_3/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_4/Assert/Assert?
(model/category_encoding_4/bincount/ShapeShape'model/string_lookup_4/Identity:output:0(^model/category_encoding_4/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_4/bincount/Shape?
(model/category_encoding_4/bincount/ConstConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_4/bincount/Const?
'model/category_encoding_4/bincount/ProdProd1model/category_encoding_4/bincount/Shape:output:01model/category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_4/bincount/Prod?
,model/category_encoding_4/bincount/Greater/yConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_4/bincount/Greater/y?
*model/category_encoding_4/bincount/GreaterGreater0model/category_encoding_4/bincount/Prod:output:05model/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_4/bincount/Greater?
'model/category_encoding_4/bincount/CastCast.model/category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_4/bincount/Cast?
*model/category_encoding_4/bincount/Const_1Const(^model/category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_4/bincount/Const_1?
&model/category_encoding_4/bincount/MaxMax'model/string_lookup_4/Identity:output:03model/category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/bincount/Max?
(model/category_encoding_4/bincount/add/yConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_4/bincount/add/y?
&model/category_encoding_4/bincount/addAddV2/model/category_encoding_4/bincount/Max:output:01model/category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/bincount/add?
&model/category_encoding_4/bincount/mulMul+model/category_encoding_4/bincount/Cast:y:0*model/category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_4/bincount/mul?
,model/category_encoding_4/bincount/minlengthConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_4/bincount/minlength?
*model/category_encoding_4/bincount/MaximumMaximum5model/category_encoding_4/bincount/minlength:output:0*model/category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_4/bincount/Maximum?
,model/category_encoding_4/bincount/maxlengthConst(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_4/bincount/maxlength?
*model/category_encoding_4/bincount/MinimumMinimum5model/category_encoding_4/bincount/maxlength:output:0.model/category_encoding_4/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_4/bincount/Minimum?
*model/category_encoding_4/bincount/Const_2Const(^model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_4/bincount/Const_2?
0model/category_encoding_4/bincount/DenseBincountDenseBincount'model/string_lookup_4/Identity:output:0.model/category_encoding_4/bincount/Minimum:z:03model/category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(22
0model/category_encoding_4/bincount/DenseBincount?
model/category_encoding_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_5/Const?
model/category_encoding_5/MaxMax'model/string_lookup_5/Identity:output:0(model/category_encoding_5/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_5/Max?
!model/category_encoding_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_5/Const_1?
model/category_encoding_5/MinMin'model/string_lookup_5/Identity:output:0*model/category_encoding_5/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_5/Min?
 model/category_encoding_5/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_5/Cast/x?
model/category_encoding_5/CastCast)model/category_encoding_5/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_5/Cast?
!model/category_encoding_5/GreaterGreater"model/category_encoding_5/Cast:y:0&model/category_encoding_5/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_5/Greater?
"model/category_encoding_5/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_5/Cast_1/x?
 model/category_encoding_5/Cast_1Cast+model/category_encoding_5/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_5/Cast_1?
&model/category_encoding_5/GreaterEqualGreaterEqual&model/category_encoding_5/Min:output:0$model/category_encoding_5/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/GreaterEqual?
$model/category_encoding_5/LogicalAnd
LogicalAnd%model/category_encoding_5/Greater:z:0*model/category_encoding_5/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_5/LogicalAnd?
&model/category_encoding_5/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952(
&model/category_encoding_5/Assert/Const?
.model/category_encoding_5/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=49520
.model/category_encoding_5/Assert/Assert/data_0?
'model/category_encoding_5/Assert/AssertAssert(model/category_encoding_5/LogicalAnd:z:07model/category_encoding_5/Assert/Assert/data_0:output:0(^model/category_encoding_4/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_5/Assert/Assert?
(model/category_encoding_5/bincount/ShapeShape'model/string_lookup_5/Identity:output:0(^model/category_encoding_5/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_5/bincount/Shape?
(model/category_encoding_5/bincount/ConstConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_5/bincount/Const?
'model/category_encoding_5/bincount/ProdProd1model/category_encoding_5/bincount/Shape:output:01model/category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_5/bincount/Prod?
,model/category_encoding_5/bincount/Greater/yConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_5/bincount/Greater/y?
*model/category_encoding_5/bincount/GreaterGreater0model/category_encoding_5/bincount/Prod:output:05model/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_5/bincount/Greater?
'model/category_encoding_5/bincount/CastCast.model/category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_5/bincount/Cast?
*model/category_encoding_5/bincount/Const_1Const(^model/category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_5/bincount/Const_1?
&model/category_encoding_5/bincount/MaxMax'model/string_lookup_5/Identity:output:03model/category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/bincount/Max?
(model/category_encoding_5/bincount/add/yConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_5/bincount/add/y?
&model/category_encoding_5/bincount/addAddV2/model/category_encoding_5/bincount/Max:output:01model/category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/bincount/add?
&model/category_encoding_5/bincount/mulMul+model/category_encoding_5/bincount/Cast:y:0*model/category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_5/bincount/mul?
,model/category_encoding_5/bincount/minlengthConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_5/bincount/minlength?
*model/category_encoding_5/bincount/MaximumMaximum5model/category_encoding_5/bincount/minlength:output:0*model/category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_5/bincount/Maximum?
,model/category_encoding_5/bincount/maxlengthConst(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_5/bincount/maxlength?
*model/category_encoding_5/bincount/MinimumMinimum5model/category_encoding_5/bincount/maxlength:output:0.model/category_encoding_5/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_5/bincount/Minimum?
*model/category_encoding_5/bincount/Const_2Const(^model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_5/bincount/Const_2?
0model/category_encoding_5/bincount/DenseBincountDenseBincount'model/string_lookup_5/Identity:output:0.model/category_encoding_5/bincount/Minimum:z:03model/category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_5/bincount/DenseBincount?
model/category_encoding_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_6/Const?
model/category_encoding_6/MaxMax'model/string_lookup_6/Identity:output:0(model/category_encoding_6/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_6/Max?
!model/category_encoding_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_6/Const_1?
model/category_encoding_6/MinMin'model/string_lookup_6/Identity:output:0*model/category_encoding_6/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_6/Min?
 model/category_encoding_6/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_6/Cast/x?
model/category_encoding_6/CastCast)model/category_encoding_6/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_6/Cast?
!model/category_encoding_6/GreaterGreater"model/category_encoding_6/Cast:y:0&model/category_encoding_6/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_6/Greater?
"model/category_encoding_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_6/Cast_1/x?
 model/category_encoding_6/Cast_1Cast+model/category_encoding_6/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_6/Cast_1?
&model/category_encoding_6/GreaterEqualGreaterEqual&model/category_encoding_6/Min:output:0$model/category_encoding_6/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/GreaterEqual?
$model/category_encoding_6/LogicalAnd
LogicalAnd%model/category_encoding_6/Greater:z:0*model/category_encoding_6/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_6/LogicalAnd?
&model/category_encoding_6/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172(
&model/category_encoding_6/Assert/Const?
.model/category_encoding_6/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=21720
.model/category_encoding_6/Assert/Assert/data_0?
'model/category_encoding_6/Assert/AssertAssert(model/category_encoding_6/LogicalAnd:z:07model/category_encoding_6/Assert/Assert/data_0:output:0(^model/category_encoding_5/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_6/Assert/Assert?
(model/category_encoding_6/bincount/ShapeShape'model/string_lookup_6/Identity:output:0(^model/category_encoding_6/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_6/bincount/Shape?
(model/category_encoding_6/bincount/ConstConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_6/bincount/Const?
'model/category_encoding_6/bincount/ProdProd1model/category_encoding_6/bincount/Shape:output:01model/category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_6/bincount/Prod?
,model/category_encoding_6/bincount/Greater/yConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_6/bincount/Greater/y?
*model/category_encoding_6/bincount/GreaterGreater0model/category_encoding_6/bincount/Prod:output:05model/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_6/bincount/Greater?
'model/category_encoding_6/bincount/CastCast.model/category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_6/bincount/Cast?
*model/category_encoding_6/bincount/Const_1Const(^model/category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_6/bincount/Const_1?
&model/category_encoding_6/bincount/MaxMax'model/string_lookup_6/Identity:output:03model/category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/bincount/Max?
(model/category_encoding_6/bincount/add/yConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_6/bincount/add/y?
&model/category_encoding_6/bincount/addAddV2/model/category_encoding_6/bincount/Max:output:01model/category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/bincount/add?
&model/category_encoding_6/bincount/mulMul+model/category_encoding_6/bincount/Cast:y:0*model/category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_6/bincount/mul?
,model/category_encoding_6/bincount/minlengthConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_6/bincount/minlength?
*model/category_encoding_6/bincount/MaximumMaximum5model/category_encoding_6/bincount/minlength:output:0*model/category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_6/bincount/Maximum?
,model/category_encoding_6/bincount/maxlengthConst(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_6/bincount/maxlength?
*model/category_encoding_6/bincount/MinimumMinimum5model/category_encoding_6/bincount/maxlength:output:0.model/category_encoding_6/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_6/bincount/Minimum?
*model/category_encoding_6/bincount/Const_2Const(^model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_6/bincount/Const_2?
0model/category_encoding_6/bincount/DenseBincountDenseBincount'model/string_lookup_6/Identity:output:0.model/category_encoding_6/bincount/Minimum:z:03model/category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_6/bincount/DenseBincount?
model/category_encoding_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_7/Const?
model/category_encoding_7/MaxMax'model/string_lookup_7/Identity:output:0(model/category_encoding_7/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_7/Max?
!model/category_encoding_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_7/Const_1?
model/category_encoding_7/MinMin'model/string_lookup_7/Identity:output:0*model/category_encoding_7/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_7/Min?
 model/category_encoding_7/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/category_encoding_7/Cast/x?
model/category_encoding_7/CastCast)model/category_encoding_7/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_7/Cast?
!model/category_encoding_7/GreaterGreater"model/category_encoding_7/Cast:y:0&model/category_encoding_7/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_7/Greater?
"model/category_encoding_7/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_7/Cast_1/x?
 model/category_encoding_7/Cast_1Cast+model/category_encoding_7/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_7/Cast_1?
&model/category_encoding_7/GreaterEqualGreaterEqual&model/category_encoding_7/Min:output:0$model/category_encoding_7/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/GreaterEqual?
$model/category_encoding_7/LogicalAnd
LogicalAnd%model/category_encoding_7/Greater:z:0*model/category_encoding_7/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_7/LogicalAnd?
&model/category_encoding_7/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442(
&model/category_encoding_7/Assert/Const?
.model/category_encoding_7/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=74420
.model/category_encoding_7/Assert/Assert/data_0?
'model/category_encoding_7/Assert/AssertAssert(model/category_encoding_7/LogicalAnd:z:07model/category_encoding_7/Assert/Assert/data_0:output:0(^model/category_encoding_6/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_7/Assert/Assert?
(model/category_encoding_7/bincount/ShapeShape'model/string_lookup_7/Identity:output:0(^model/category_encoding_7/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_7/bincount/Shape?
(model/category_encoding_7/bincount/ConstConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_7/bincount/Const?
'model/category_encoding_7/bincount/ProdProd1model/category_encoding_7/bincount/Shape:output:01model/category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_7/bincount/Prod?
,model/category_encoding_7/bincount/Greater/yConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_7/bincount/Greater/y?
*model/category_encoding_7/bincount/GreaterGreater0model/category_encoding_7/bincount/Prod:output:05model/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_7/bincount/Greater?
'model/category_encoding_7/bincount/CastCast.model/category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_7/bincount/Cast?
*model/category_encoding_7/bincount/Const_1Const(^model/category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_7/bincount/Const_1?
&model/category_encoding_7/bincount/MaxMax'model/string_lookup_7/Identity:output:03model/category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/bincount/Max?
(model/category_encoding_7/bincount/add/yConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_7/bincount/add/y?
&model/category_encoding_7/bincount/addAddV2/model/category_encoding_7/bincount/Max:output:01model/category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/bincount/add?
&model/category_encoding_7/bincount/mulMul+model/category_encoding_7/bincount/Cast:y:0*model/category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_7/bincount/mul?
,model/category_encoding_7/bincount/minlengthConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_7/bincount/minlength?
*model/category_encoding_7/bincount/MaximumMaximum5model/category_encoding_7/bincount/minlength:output:0*model/category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_7/bincount/Maximum?
,model/category_encoding_7/bincount/maxlengthConst(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2.
,model/category_encoding_7/bincount/maxlength?
*model/category_encoding_7/bincount/MinimumMinimum5model/category_encoding_7/bincount/maxlength:output:0.model/category_encoding_7/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_7/bincount/Minimum?
*model/category_encoding_7/bincount/Const_2Const(^model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_7/bincount/Const_2?
0model/category_encoding_7/bincount/DenseBincountDenseBincount'model/string_lookup_7/Identity:output:0.model/category_encoding_7/bincount/Minimum:z:03model/category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(22
0model/category_encoding_7/bincount/DenseBincount?
model/category_encoding_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/category_encoding_8/Const?
model/category_encoding_8/MaxMax'model/string_lookup_8/Identity:output:0(model/category_encoding_8/Const:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_8/Max?
!model/category_encoding_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/category_encoding_8/Const_1?
model/category_encoding_8/MinMin'model/string_lookup_8/Identity:output:0*model/category_encoding_8/Const_1:output:0*
T0	*
_output_shapes
: 2
model/category_encoding_8/Min?
 model/category_encoding_8/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/category_encoding_8/Cast/x?
model/category_encoding_8/CastCast)model/category_encoding_8/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2 
model/category_encoding_8/Cast?
!model/category_encoding_8/GreaterGreater"model/category_encoding_8/Cast:y:0&model/category_encoding_8/Max:output:0*
T0	*
_output_shapes
: 2#
!model/category_encoding_8/Greater?
"model/category_encoding_8/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model/category_encoding_8/Cast_1/x?
 model/category_encoding_8/Cast_1Cast+model/category_encoding_8/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2"
 model/category_encoding_8/Cast_1?
&model/category_encoding_8/GreaterEqualGreaterEqual&model/category_encoding_8/Min:output:0$model/category_encoding_8/Cast_1:y:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/GreaterEqual?
$model/category_encoding_8/LogicalAnd
LogicalAnd%model/category_encoding_8/Greater:z:0*model/category_encoding_8/GreaterEqual:z:0*
_output_shapes
: 2&
$model/category_encoding_8/LogicalAnd?
&model/category_encoding_8/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112(
&model/category_encoding_8/Assert/Const?
.model/category_encoding_8/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=1120
.model/category_encoding_8/Assert/Assert/data_0?
'model/category_encoding_8/Assert/AssertAssert(model/category_encoding_8/LogicalAnd:z:07model/category_encoding_8/Assert/Assert/data_0:output:0(^model/category_encoding_7/Assert/Assert*

T
2*
_output_shapes
 2)
'model/category_encoding_8/Assert/Assert?
(model/category_encoding_8/bincount/ShapeShape'model/string_lookup_8/Identity:output:0(^model/category_encoding_8/Assert/Assert*
T0	*
_output_shapes
:2*
(model/category_encoding_8/bincount/Shape?
(model/category_encoding_8/bincount/ConstConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2*
(model/category_encoding_8/bincount/Const?
'model/category_encoding_8/bincount/ProdProd1model/category_encoding_8/bincount/Shape:output:01model/category_encoding_8/bincount/Const:output:0*
T0*
_output_shapes
: 2)
'model/category_encoding_8/bincount/Prod?
,model/category_encoding_8/bincount/Greater/yConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2.
,model/category_encoding_8/bincount/Greater/y?
*model/category_encoding_8/bincount/GreaterGreater0model/category_encoding_8/bincount/Prod:output:05model/category_encoding_8/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2,
*model/category_encoding_8/bincount/Greater?
'model/category_encoding_8/bincount/CastCast.model/category_encoding_8/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2)
'model/category_encoding_8/bincount/Cast?
*model/category_encoding_8/bincount/Const_1Const(^model/category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2,
*model/category_encoding_8/bincount/Const_1?
&model/category_encoding_8/bincount/MaxMax'model/string_lookup_8/Identity:output:03model/category_encoding_8/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/bincount/Max?
(model/category_encoding_8/bincount/add/yConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2*
(model/category_encoding_8/bincount/add/y?
&model/category_encoding_8/bincount/addAddV2/model/category_encoding_8/bincount/Max:output:01model/category_encoding_8/bincount/add/y:output:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/bincount/add?
&model/category_encoding_8/bincount/mulMul+model/category_encoding_8/bincount/Cast:y:0*model/category_encoding_8/bincount/add:z:0*
T0	*
_output_shapes
: 2(
&model/category_encoding_8/bincount/mul?
,model/category_encoding_8/bincount/minlengthConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_8/bincount/minlength?
*model/category_encoding_8/bincount/MaximumMaximum5model/category_encoding_8/bincount/minlength:output:0*model/category_encoding_8/bincount/mul:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_8/bincount/Maximum?
,model/category_encoding_8/bincount/maxlengthConst(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2.
,model/category_encoding_8/bincount/maxlength?
*model/category_encoding_8/bincount/MinimumMinimum5model/category_encoding_8/bincount/maxlength:output:0.model/category_encoding_8/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2,
*model/category_encoding_8/bincount/Minimum?
*model/category_encoding_8/bincount/Const_2Const(^model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2,
*model/category_encoding_8/bincount/Const_2?
0model/category_encoding_8/bincount/DenseBincountDenseBincount'model/string_lookup_8/Identity:output:0.model/category_encoding_8/bincount/Minimum:z:03model/category_encoding_8/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(22
0model/category_encoding_8/bincount/DenseBincount?
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_1/concat/axis?
model/concatenate_1/concatConcatV2model/normalization/truediv:z:07model/category_encoding/bincount/DenseBincount:output:09model/category_encoding_1/bincount/DenseBincount:output:09model/category_encoding_2/bincount/DenseBincount:output:09model/category_encoding_3/bincount/DenseBincount:output:09model/category_encoding_4/bincount/DenseBincount:output:09model/category_encoding_5/bincount/DenseBincount:output:09model/category_encoding_6/bincount/DenseBincount:output:09model/category_encoding_7/bincount/DenseBincount:output:09model/category_encoding_8/bincount/DenseBincount:output:0(model/concatenate_1/concat/axis:output:0*
N
*
T0*(
_output_shapes
:?????????? 2
model/concatenate_1/concat?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#model/concatenate_1/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/BiasAdd?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul!sequential/dense/BiasAdd:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd~
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^model/category_encoding/Assert/Assert(^model/category_encoding_1/Assert/Assert(^model/category_encoding_2/Assert/Assert(^model/category_encoding_3/Assert/Assert(^model/category_encoding_4/Assert/Assert(^model/category_encoding_5/Assert/Assert(^model/category_encoding_6/Assert/Assert(^model/category_encoding_7/Assert/Assert(^model/category_encoding_8/Assert/Assert2^model/string_lookup/None_Lookup/LookupTableFindV24^model/string_lookup_1/None_Lookup/LookupTableFindV24^model/string_lookup_2/None_Lookup/LookupTableFindV24^model/string_lookup_3/None_Lookup/LookupTableFindV24^model/string_lookup_4/None_Lookup/LookupTableFindV24^model/string_lookup_5/None_Lookup/LookupTableFindV24^model/string_lookup_6/None_Lookup/LookupTableFindV24^model/string_lookup_7/None_Lookup/LookupTableFindV24^model/string_lookup_8/None_Lookup/LookupTableFindV2(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 2N
%model/category_encoding/Assert/Assert%model/category_encoding/Assert/Assert2R
'model/category_encoding_1/Assert/Assert'model/category_encoding_1/Assert/Assert2R
'model/category_encoding_2/Assert/Assert'model/category_encoding_2/Assert/Assert2R
'model/category_encoding_3/Assert/Assert'model/category_encoding_3/Assert/Assert2R
'model/category_encoding_4/Assert/Assert'model/category_encoding_4/Assert/Assert2R
'model/category_encoding_5/Assert/Assert'model/category_encoding_5/Assert/Assert2R
'model/category_encoding_6/Assert/Assert'model/category_encoding_6/Assert/Assert2R
'model/category_encoding_7/Assert/Assert'model/category_encoding_7/Assert/Assert2R
'model/category_encoding_8/Assert/Assert'model/category_encoding_8/Assert/Assert2f
1model/string_lookup/None_Lookup/LookupTableFindV21model/string_lookup/None_Lookup/LookupTableFindV22j
3model/string_lookup_1/None_Lookup/LookupTableFindV23model/string_lookup_1/None_Lookup/LookupTableFindV22j
3model/string_lookup_2/None_Lookup/LookupTableFindV23model/string_lookup_2/None_Lookup/LookupTableFindV22j
3model/string_lookup_3/None_Lookup/LookupTableFindV23model/string_lookup_3/None_Lookup/LookupTableFindV22j
3model/string_lookup_4/None_Lookup/LookupTableFindV23model/string_lookup_4/None_Lookup/LookupTableFindV22j
3model/string_lookup_5/None_Lookup/LookupTableFindV23model/string_lookup_5/None_Lookup/LookupTableFindV22j
3model/string_lookup_6/None_Lookup/LookupTableFindV23model/string_lookup_6/None_Lookup/LookupTableFindV22j
3model/string_lookup_7/None_Lookup/LookupTableFindV23model/string_lookup_7/None_Lookup/LookupTableFindV22j
3model/string_lookup_8/None_Lookup/LookupTableFindV23model/string_lookup_8/None_Lookup/LookupTableFindV22R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
?
__inference__initializer_137557
3key_value_init6626_lookuptableimportv2_table_handle/
+key_value_init6626_lookuptableimportv2_keys1
-key_value_init6626_lookuptableimportv2_values	
identity??&key_value_init6626/LookupTableImportV2?
&key_value_init6626/LookupTableImportV2LookupTableImportV23key_value_init6626_lookuptableimportv2_table_handle+key_value_init6626_lookuptableimportv2_keys-key_value_init6626_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6626/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6626/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6626/LookupTableImportV2&key_value_init6626/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
? 
|
M__inference_category_encoding_1_layer_call_and_return_conditional_losses_9894

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_138587
3key_value_init6470_lookuptableimportv2_table_handle/
+key_value_init6470_lookuptableimportv2_keys1
-key_value_init6470_lookuptableimportv2_values	
identity??&key_value_init6470/LookupTableImportV2?
&key_value_init6470/LookupTableImportV2LookupTableImportV23key_value_init6470_lookuptableimportv2_table_handle+key_value_init6470_lookuptableimportv2_keys-key_value_init6470_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6470/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6470/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6470/LookupTableImportV2&key_value_init6470/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
:
__inference__creator_13729
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6575*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_10803

inputs
dense_10792:	? @
dense_10794:@
dense_1_10797:@
dense_1_10799:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_10792dense_10794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_107202
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_10797dense_1_10799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_107362!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
,
__inference__destroyer_13760
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
:
__inference__creator_13747
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6627*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
? 
?
%__inference_model_layer_call_fn_10553
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallachievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_104482
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
?
__inference__initializer_138277
3key_value_init6834_lookuptableimportv2_table_handle/
+key_value_init6834_lookuptableimportv2_keys1
-key_value_init6834_lookuptableimportv2_values	
identity??&key_value_init6834/LookupTableImportV2?
&key_value_init6834/LookupTableImportV2LookupTableImportV23key_value_init6834_lookuptableimportv2_table_handle+key_value_init6834_lookuptableimportv2_keys-key_value_init6834_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6834/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6834/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6834/LookupTableImportV2&key_value_init6834/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
__inference_<lambda>_138907
3key_value_init6678_lookuptableimportv2_table_handle/
+key_value_init6678_lookuptableimportv2_keys1
-key_value_init6678_lookuptableimportv2_values	
identity??&key_value_init6678/LookupTableImportV2?
&key_value_init6678/LookupTableImportV2LookupTableImportV23key_value_init6678_lookuptableimportv2_table_handle+key_value_init6678_lookuptableimportv2_keys-key_value_init6678_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6678/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6678/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2P
&key_value_init6678/LookupTableImportV2&key_value_init6678/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?!
?
'__inference_model_1_layer_call_fn_10998
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18

unknown_19:	? @

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallachievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*5
Tin.
,2*									*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
&'()*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_109472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?&
?
B__inference_model_1_layer_call_and_return_conditional_losses_11158

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
model_11107
model_11109	
model_11111
model_11113	
model_11115
model_11117	
model_11119
model_11121	
model_11123
model_11125	
model_11127
model_11129	
model_11131
model_11133	
model_11135
model_11137	
model_11139
model_11141	
model_11143
model_11145#
sequential_11148:	? @
sequential_11150:@"
sequential_11152:@
sequential_11154:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17model_11107model_11109model_11111model_11113model_11115model_11117model_11119model_11121model_11123model_11125model_11127model_11129model_11131model_11133model_11135model_11137model_11139model_11141model_11143model_11145*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_104482
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_11148sequential_11150sequential_11152sequential_11154*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_108032$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_13679

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_13155

inputs7
$dense_matmul_readvariableop_resource:	? @3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
__inference__initializer_137377
3key_value_init6574_lookuptableimportv2_table_handle/
+key_value_init6574_lookuptableimportv2_keys1
-key_value_init6574_lookuptableimportv2_values	
identity??&key_value_init6574/LookupTableImportV2?
&key_value_init6574/LookupTableImportV2LookupTableImportV23key_value_init6574_lookuptableimportv2_table_handle+key_value_init6574_lookuptableimportv2_keys-key_value_init6574_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6574/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6574/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6574/LookupTableImportV2&key_value_init6574/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
? 
}
N__inference_category_encoding_3_layer_call_and_return_conditional_losses_13421

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
}
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_10146

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinR
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mulz
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximumz
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2
bincount/DenseBincountz
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
}
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_13577

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_10855
dense_input
dense_10844:	? @
dense_10846:@
dense_1_10849:@
dense_1_10851:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_10844dense_10846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_107202
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_10849dense_1_10851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_107362!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:U Q
(
_output_shapes
:?????????? 
%
_user_specified_namedense_input
?
?
__inference_<lambda>_139147
3key_value_init6834_lookuptableimportv2_table_handle/
+key_value_init6834_lookuptableimportv2_keys1
-key_value_init6834_lookuptableimportv2_values	
identity??&key_value_init6834/LookupTableImportV2?
&key_value_init6834/LookupTableImportV2LookupTableImportV23key_value_init6834_lookuptableimportv2_table_handle+key_value_init6834_lookuptableimportv2_keys-key_value_init6834_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6834/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6834/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6834/LookupTableImportV2&key_value_init6834/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_10841
dense_input
dense_10830:	? @
dense_10832:@
dense_1_10835:@
dense_1_10837:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_10830dense_10832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_107202
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_10835dense_1_10837*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_107362!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:U Q
(
_output_shapes
:?????????? 
%
_user_specified_namedense_input
?
?
*__inference_sequential_layer_call_fn_10754
dense_input
unknown:	? @
	unknown_0:@
	unknown_1:@
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_107432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:?????????? 
%
_user_specified_namedense_input
? 
}
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_13616

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinR
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mulz
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximumz
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2
bincount/DenseBincountz
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_13778
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
,
__inference__destroyer_13832
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?#
?
%__inference_model_layer_call_fn_13139
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_achievementsinputs_appidinputs_average_playtimeinputs_categoriesinputs_developerinputs_englishinputs_genresinputs_median_playtimeinputs_nameinputs_negative_ratingsinputs_ownersinputs_platformsinputs_positive_ratingsinputs_priceinputs_publisherinputs_release_dateinputs_required_ageinputs_steamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_104482
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
?
__inference_<lambda>_139067
3key_value_init6782_lookuptableimportv2_table_handle/
+key_value_init6782_lookuptableimportv2_keys1
-key_value_init6782_lookuptableimportv2_values	
identity??&key_value_init6782/LookupTableImportV2?
&key_value_init6782/LookupTableImportV2LookupTableImportV23key_value_init6782_lookuptableimportv2_table_handle+key_value_init6782_lookuptableimportv2_keys-key_value_init6782_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6782/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6782/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6782/LookupTableImportV2&key_value_init6782/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
__inference_<lambda>_138667
3key_value_init6522_lookuptableimportv2_table_handle/
+key_value_init6522_lookuptableimportv2_keys1
-key_value_init6522_lookuptableimportv2_values	
identity??&key_value_init6522/LookupTableImportV2?
&key_value_init6522/LookupTableImportV2LookupTableImportV23key_value_init6522_lookuptableimportv2_table_handle+key_value_init6522_lookuptableimportv2_keys-key_value_init6522_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6522/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6522/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6522/LookupTableImportV2&key_value_init6522/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
ތ
?
@__inference_model_layer_call_and_return_conditional_losses_10166

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17>
:string_lookup_8_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_4_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_4_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
identity??)category_encoding/StatefulPartitionedCall?+category_encoding_1/StatefulPartitionedCall?+category_encoding_2/StatefulPartitionedCall?+category_encoding_3/StatefulPartitionedCall?+category_encoding_4/StatefulPartitionedCall?+category_encoding_5/StatefulPartitionedCall?+category_encoding_6/StatefulPartitionedCall?+category_encoding_7/StatefulPartitionedCall?+category_encoding_8/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?-string_lookup_1/None_Lookup/LookupTableFindV2?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2?-string_lookup_4/None_Lookup/LookupTableFindV2?-string_lookup_5/None_Lookup/LookupTableFindV2?-string_lookup_6/None_Lookup/LookupTableFindV2?-string_lookup_7/None_Lookup/LookupTableFindV2?-string_lookup_8/None_Lookup/LookupTableFindV2?
-string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_8_none_lookup_lookuptablefindv2_table_handle	inputs_10;string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_8/None_Lookup/LookupTableFindV2?
string_lookup_8/IdentityIdentity6string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_8/Identity?
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handle	inputs_17;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_7/None_Lookup/LookupTableFindV2?
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_7/Identity?
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handleinputs_6;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_6/None_Lookup/LookupTableFindV2?
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_6/Identity?
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handleinputs_3;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_5/None_Lookup/LookupTableFindV2?
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_5/Identity?
-string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_4_none_lookup_lookuptablefindv2_table_handle	inputs_11;string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_4/None_Lookup/LookupTableFindV2?
string_lookup_4/IdentityIdentity6string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_4/Identity?
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handle	inputs_14;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_3/None_Lookup/LookupTableFindV2?
string_lookup_3/IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_3/Identity?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_4;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_2/Identity?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handle	inputs_15;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_89string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
concatenate/PartitionedCallPartitionedCallinputs_1inputs_5	inputs_16inputs	inputs_12inputs_9inputs_2inputs_7	inputs_13*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_98152
concatenate/PartitionedCall?
normalization/subSub$concatenate/PartitionedCall:output:0normalization_sub_y*
T0*'
_output_shapes
:?????????	2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:	2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
normalization/truediv?
)category_encoding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_category_encoding_layer_call_and_return_conditional_losses_98582+
)category_encoding/StatefulPartitionedCall?
+category_encoding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0*^category_encoding/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_1_layer_call_and_return_conditional_losses_98942-
+category_encoding_1/StatefulPartitionedCall?
+category_encoding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0,^category_encoding_1/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_2_layer_call_and_return_conditional_losses_99302-
+category_encoding_2/StatefulPartitionedCall?
+category_encoding_3/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_3/Identity:output:0,^category_encoding_2/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_3_layer_call_and_return_conditional_losses_99662-
+category_encoding_3/StatefulPartitionedCall?
+category_encoding_4/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_4/Identity:output:0,^category_encoding_3/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_100022-
+category_encoding_4/StatefulPartitionedCall?
+category_encoding_5/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_5/Identity:output:0,^category_encoding_4/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_100382-
+category_encoding_5/StatefulPartitionedCall?
+category_encoding_6/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_6/Identity:output:0,^category_encoding_5/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_100742-
+category_encoding_6/StatefulPartitionedCall?
+category_encoding_7/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_7/Identity:output:0,^category_encoding_6/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_101102-
+category_encoding_7/StatefulPartitionedCall?
+category_encoding_8/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_8/Identity:output:0,^category_encoding_7/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_101462-
+category_encoding_8/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:02category_encoding/StatefulPartitionedCall:output:04category_encoding_1/StatefulPartitionedCall:output:04category_encoding_2/StatefulPartitionedCall:output:04category_encoding_3/StatefulPartitionedCall:output:04category_encoding_4/StatefulPartitionedCall:output:04category_encoding_5/StatefulPartitionedCall:output:04category_encoding_6/StatefulPartitionedCall:output:04category_encoding_7/StatefulPartitionedCall:output:04category_encoding_8/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_101632
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identity?
NoOpNoOp*^category_encoding/StatefulPartitionedCall,^category_encoding_1/StatefulPartitionedCall,^category_encoding_2/StatefulPartitionedCall,^category_encoding_3/StatefulPartitionedCall,^category_encoding_4/StatefulPartitionedCall,^category_encoding_5/StatefulPartitionedCall,^category_encoding_6/StatefulPartitionedCall,^category_encoding_7/StatefulPartitionedCall,^category_encoding_8/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2.^string_lookup_1/None_Lookup/LookupTableFindV2.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2.^string_lookup_4/None_Lookup/LookupTableFindV2.^string_lookup_5/None_Lookup/LookupTableFindV2.^string_lookup_6/None_Lookup/LookupTableFindV2.^string_lookup_7/None_Lookup/LookupTableFindV2.^string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	2V
)category_encoding/StatefulPartitionedCall)category_encoding/StatefulPartitionedCall2Z
+category_encoding_1/StatefulPartitionedCall+category_encoding_1/StatefulPartitionedCall2Z
+category_encoding_2/StatefulPartitionedCall+category_encoding_2/StatefulPartitionedCall2Z
+category_encoding_3/StatefulPartitionedCall+category_encoding_3/StatefulPartitionedCall2Z
+category_encoding_4/StatefulPartitionedCall+category_encoding_4/StatefulPartitionedCall2Z
+category_encoding_5/StatefulPartitionedCall+category_encoding_5/StatefulPartitionedCall2Z
+category_encoding_6/StatefulPartitionedCall+category_encoding_6/StatefulPartitionedCall2Z
+category_encoding_7/StatefulPartitionedCall+category_encoding_7/StatefulPartitionedCall2Z
+category_encoding_8/StatefulPartitionedCall+category_encoding_8/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV22^
-string_lookup_4/None_Lookup/LookupTableFindV2-string_lookup_4/None_Lookup/LookupTableFindV22^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV22^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV22^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV22^
-string_lookup_8/None_Lookup/LookupTableFindV2-string_lookup_8/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_13171

inputs7
$dense_matmul_readvariableop_resource:	? @3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?!
?
'__inference_model_1_layer_call_fn_11279
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18

unknown_19:	? @

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallachievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*5
Tin.
,2*									*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
&'()*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_111582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
,
__inference__destroyer_13814
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
? 
}
N__inference_category_encoding_1_layer_call_and_return_conditional_losses_13343

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
3__inference_category_encoding_2_layer_call_fn_13387

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_2_layer_call_and_return_conditional_losses_99302
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__initializer_137737
3key_value_init6678_lookuptableimportv2_table_handle/
+key_value_init6678_lookuptableimportv2_keys1
-key_value_init6678_lookuptableimportv2_values	
identity??&key_value_init6678/LookupTableImportV2?
&key_value_init6678/LookupTableImportV2LookupTableImportV23key_value_init6678_lookuptableimportv2_table_handle+key_value_init6678_lookuptableimportv2_keys-key_value_init6678_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6678/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6678/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2P
&key_value_init6678/LookupTableImportV2&key_value_init6678/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
%__inference_dense_layer_call_fn_13669

inputs
unknown:	? @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_107202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
? 
}
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_10002

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinR
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mulz
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximumz
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2
bincount/DenseBincountz
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
}
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_10074

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
:
__inference__creator_13765
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6679*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
? 
?
%__inference_model_layer_call_fn_10209
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallachievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_101662
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	22
StatefulPartitionedCallStatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?]
?
!__inference__traced_restore_14151
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: %
assignvariableop_5_mean:	)
assignvariableop_6_variance:	"
assignvariableop_7_count:	 2
assignvariableop_8_dense_kernel:	? @+
assignvariableop_9_dense_bias:@4
"assignvariableop_10_dense_1_kernel:@.
 assignvariableop_11_dense_1_bias:#
assignvariableop_12_total: %
assignvariableop_13_count_1: :
'assignvariableop_14_adam_dense_kernel_m:	? @3
%assignvariableop_15_adam_dense_bias_m:@;
)assignvariableop_16_adam_dense_1_kernel_m:@5
'assignvariableop_17_adam_dense_1_bias_m::
'assignvariableop_18_adam_dense_kernel_v:	? @3
%assignvariableop_19_adam_dense_bias_v:@;
)assignvariableop_20_adam_dense_1_kernel_v:@5
'assignvariableop_21_adam_dense_1_bias_v:
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_meanIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_varianceIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_adam_dense_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_1_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_dense_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_1_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_1_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22f
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_23?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
? 
{
L__inference_category_encoding_layer_call_and_return_conditional_losses_13304

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
z
K__inference_category_encoding_layer_call_and_return_conditional_losses_9858

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
3__inference_category_encoding_8_layer_call_fn_13621

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_101462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
|
M__inference_category_encoding_3_layer_call_and_return_conditional_losses_9966

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_13688

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_107362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
l
3__inference_category_encoding_7_layer_call_fn_13582

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_101102
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
__inference_adapt_step_13270
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:	'
readvariableop_2_resource:	??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????	*&
output_shapes
:?????????	*
output_types
22
IteratorGetNext?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:	*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:	2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????	2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:	*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 2
moments/Squeeze_1j
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	2
Shapen
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: 2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addS
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
CastQ
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1T
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:	2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:	2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:	2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:	2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:	2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:	*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:	2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:	2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:	2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:	2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:	2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:	2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:	2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
? 
}
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_10038

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_10628
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags>
:string_lookup_8_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_4_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_4_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
identity??)category_encoding/StatefulPartitionedCall?+category_encoding_1/StatefulPartitionedCall?+category_encoding_2/StatefulPartitionedCall?+category_encoding_3/StatefulPartitionedCall?+category_encoding_4/StatefulPartitionedCall?+category_encoding_5/StatefulPartitionedCall?+category_encoding_6/StatefulPartitionedCall?+category_encoding_7/StatefulPartitionedCall?+category_encoding_8/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?-string_lookup_1/None_Lookup/LookupTableFindV2?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2?-string_lookup_4/None_Lookup/LookupTableFindV2?-string_lookup_5/None_Lookup/LookupTableFindV2?-string_lookup_6/None_Lookup/LookupTableFindV2?-string_lookup_7/None_Lookup/LookupTableFindV2?-string_lookup_8/None_Lookup/LookupTableFindV2?
-string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_8_none_lookup_lookuptablefindv2_table_handleowners;string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_8/None_Lookup/LookupTableFindV2?
string_lookup_8/IdentityIdentity6string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_8/Identity?
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handlesteamspy_tags;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_7/None_Lookup/LookupTableFindV2?
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_7/Identity?
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handlegenres;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_6/None_Lookup/LookupTableFindV2?
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_6/Identity?
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handle
categories;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_5/None_Lookup/LookupTableFindV2?
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_5/Identity?
-string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_4_none_lookup_lookuptablefindv2_table_handle	platforms;string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_4/None_Lookup/LookupTableFindV2?
string_lookup_4/IdentityIdentity6string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_4/Identity?
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handle	publisher;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_3/None_Lookup/LookupTableFindV2?
string_lookup_3/IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_3/Identity?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handle	developer;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_2/Identity?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handlerelease_date;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlename9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
concatenate/PartitionedCallPartitionedCallappidenglishrequired_ageachievementspositive_ratingsnegative_ratingsaverage_playtimemedian_playtimeprice*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_98152
concatenate/PartitionedCall?
normalization/subSub$concatenate/PartitionedCall:output:0normalization_sub_y*
T0*'
_output_shapes
:?????????	2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:	2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
normalization/truediv?
)category_encoding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_category_encoding_layer_call_and_return_conditional_losses_98582+
)category_encoding/StatefulPartitionedCall?
+category_encoding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0*^category_encoding/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_1_layer_call_and_return_conditional_losses_98942-
+category_encoding_1/StatefulPartitionedCall?
+category_encoding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0,^category_encoding_1/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_2_layer_call_and_return_conditional_losses_99302-
+category_encoding_2/StatefulPartitionedCall?
+category_encoding_3/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_3/Identity:output:0,^category_encoding_2/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_3_layer_call_and_return_conditional_losses_99662-
+category_encoding_3/StatefulPartitionedCall?
+category_encoding_4/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_4/Identity:output:0,^category_encoding_3/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_100022-
+category_encoding_4/StatefulPartitionedCall?
+category_encoding_5/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_5/Identity:output:0,^category_encoding_4/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_100382-
+category_encoding_5/StatefulPartitionedCall?
+category_encoding_6/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_6/Identity:output:0,^category_encoding_5/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_100742-
+category_encoding_6/StatefulPartitionedCall?
+category_encoding_7/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_7/Identity:output:0,^category_encoding_6/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_101102-
+category_encoding_7/StatefulPartitionedCall?
+category_encoding_8/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_8/Identity:output:0,^category_encoding_7/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_101462-
+category_encoding_8/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:02category_encoding/StatefulPartitionedCall:output:04category_encoding_1/StatefulPartitionedCall:output:04category_encoding_2/StatefulPartitionedCall:output:04category_encoding_3/StatefulPartitionedCall:output:04category_encoding_4/StatefulPartitionedCall:output:04category_encoding_5/StatefulPartitionedCall:output:04category_encoding_6/StatefulPartitionedCall:output:04category_encoding_7/StatefulPartitionedCall:output:04category_encoding_8/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_101632
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identity?
NoOpNoOp*^category_encoding/StatefulPartitionedCall,^category_encoding_1/StatefulPartitionedCall,^category_encoding_2/StatefulPartitionedCall,^category_encoding_3/StatefulPartitionedCall,^category_encoding_4/StatefulPartitionedCall,^category_encoding_5/StatefulPartitionedCall,^category_encoding_6/StatefulPartitionedCall,^category_encoding_7/StatefulPartitionedCall,^category_encoding_8/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2.^string_lookup_1/None_Lookup/LookupTableFindV2.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2.^string_lookup_4/None_Lookup/LookupTableFindV2.^string_lookup_5/None_Lookup/LookupTableFindV2.^string_lookup_6/None_Lookup/LookupTableFindV2.^string_lookup_7/None_Lookup/LookupTableFindV2.^string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	2V
)category_encoding/StatefulPartitionedCall)category_encoding/StatefulPartitionedCall2Z
+category_encoding_1/StatefulPartitionedCall+category_encoding_1/StatefulPartitionedCall2Z
+category_encoding_2/StatefulPartitionedCall+category_encoding_2/StatefulPartitionedCall2Z
+category_encoding_3/StatefulPartitionedCall+category_encoding_3/StatefulPartitionedCall2Z
+category_encoding_4/StatefulPartitionedCall+category_encoding_4/StatefulPartitionedCall2Z
+category_encoding_5/StatefulPartitionedCall+category_encoding_5/StatefulPartitionedCall2Z
+category_encoding_6/StatefulPartitionedCall+category_encoding_6/StatefulPartitionedCall2Z
+category_encoding_7/StatefulPartitionedCall+category_encoding_7/StatefulPartitionedCall2Z
+category_encoding_8/StatefulPartitionedCall+category_encoding_8/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV22^
-string_lookup_4/None_Lookup/LookupTableFindV2-string_lookup_4/None_Lookup/LookupTableFindV22^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV22^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV22^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV22^
-string_lookup_8/None_Lookup/LookupTableFindV2-string_lookup_8/None_Lookup/LookupTableFindV2:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
?
__inference__initializer_138457
3key_value_init6886_lookuptableimportv2_table_handle/
+key_value_init6886_lookuptableimportv2_keys1
-key_value_init6886_lookuptableimportv2_values	
identity??&key_value_init6886/LookupTableImportV2?
&key_value_init6886/LookupTableImportV2LookupTableImportV23key_value_init6886_lookuptableimportv2_table_handle+key_value_init6886_lookuptableimportv2_keys-key_value_init6886_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6886/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6886/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :
:
2P
&key_value_init6886/LookupTableImportV2&key_value_init6886/LookupTableImportV2: 

_output_shapes
:
: 

_output_shapes
:

?
,
__inference__destroyer_13742
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
:
__inference__creator_13837
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6887*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?'
?
B__inference_model_1_layer_call_and_return_conditional_losses_11350
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tags
model_11299
model_11301	
model_11303
model_11305	
model_11307
model_11309	
model_11311
model_11313	
model_11315
model_11317	
model_11319
model_11321	
model_11323
model_11325	
model_11327
model_11329	
model_11331
model_11333	
model_11335
model_11337#
sequential_11340:	? @
sequential_11342:@"
sequential_11344:@
sequential_11346:
identity??model/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallachievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tagsmodel_11299model_11301model_11303model_11305model_11307model_11309model_11311model_11313model_11315model_11317model_11319model_11321model_11323model_11325model_11327model_11329model_11331model_11333model_11335model_11337*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_101662
model/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0sequential_11340sequential_11342sequential_11344sequential_11346*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_107432$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^model/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
ތ
?
@__inference_model_layer_call_and_return_conditional_losses_10448

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17>
:string_lookup_8_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_4_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_4_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
identity??)category_encoding/StatefulPartitionedCall?+category_encoding_1/StatefulPartitionedCall?+category_encoding_2/StatefulPartitionedCall?+category_encoding_3/StatefulPartitionedCall?+category_encoding_4/StatefulPartitionedCall?+category_encoding_5/StatefulPartitionedCall?+category_encoding_6/StatefulPartitionedCall?+category_encoding_7/StatefulPartitionedCall?+category_encoding_8/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?-string_lookup_1/None_Lookup/LookupTableFindV2?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2?-string_lookup_4/None_Lookup/LookupTableFindV2?-string_lookup_5/None_Lookup/LookupTableFindV2?-string_lookup_6/None_Lookup/LookupTableFindV2?-string_lookup_7/None_Lookup/LookupTableFindV2?-string_lookup_8/None_Lookup/LookupTableFindV2?
-string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_8_none_lookup_lookuptablefindv2_table_handle	inputs_10;string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_8/None_Lookup/LookupTableFindV2?
string_lookup_8/IdentityIdentity6string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_8/Identity?
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handle	inputs_17;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_7/None_Lookup/LookupTableFindV2?
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_7/Identity?
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handleinputs_6;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_6/None_Lookup/LookupTableFindV2?
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_6/Identity?
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handleinputs_3;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_5/None_Lookup/LookupTableFindV2?
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_5/Identity?
-string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_4_none_lookup_lookuptablefindv2_table_handle	inputs_11;string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_4/None_Lookup/LookupTableFindV2?
string_lookup_4/IdentityIdentity6string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_4/Identity?
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handle	inputs_14;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_3/None_Lookup/LookupTableFindV2?
string_lookup_3/IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_3/Identity?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_4;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_2/Identity?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handle	inputs_15;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_89string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identity?
concatenate/PartitionedCallPartitionedCallinputs_1inputs_5	inputs_16inputs	inputs_12inputs_9inputs_2inputs_7	inputs_13*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_98152
concatenate/PartitionedCall?
normalization/subSub$concatenate/PartitionedCall:output:0normalization_sub_y*
T0*'
_output_shapes
:?????????	2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:	2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
normalization/truediv?
)category_encoding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_category_encoding_layer_call_and_return_conditional_losses_98582+
)category_encoding/StatefulPartitionedCall?
+category_encoding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0*^category_encoding/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_1_layer_call_and_return_conditional_losses_98942-
+category_encoding_1/StatefulPartitionedCall?
+category_encoding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0,^category_encoding_1/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_2_layer_call_and_return_conditional_losses_99302-
+category_encoding_2/StatefulPartitionedCall?
+category_encoding_3/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_3/Identity:output:0,^category_encoding_2/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_category_encoding_3_layer_call_and_return_conditional_losses_99662-
+category_encoding_3/StatefulPartitionedCall?
+category_encoding_4/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_4/Identity:output:0,^category_encoding_3/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_100022-
+category_encoding_4/StatefulPartitionedCall?
+category_encoding_5/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_5/Identity:output:0,^category_encoding_4/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_100382-
+category_encoding_5/StatefulPartitionedCall?
+category_encoding_6/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_6/Identity:output:0,^category_encoding_5/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_100742-
+category_encoding_6/StatefulPartitionedCall?
+category_encoding_7/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_7/Identity:output:0,^category_encoding_6/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_101102-
+category_encoding_7/StatefulPartitionedCall?
+category_encoding_8/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_8/Identity:output:0,^category_encoding_7/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_101462-
+category_encoding_8/StatefulPartitionedCall?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:02category_encoding/StatefulPartitionedCall:output:04category_encoding_1/StatefulPartitionedCall:output:04category_encoding_2/StatefulPartitionedCall:output:04category_encoding_3/StatefulPartitionedCall:output:04category_encoding_4/StatefulPartitionedCall:output:04category_encoding_5/StatefulPartitionedCall:output:04category_encoding_6/StatefulPartitionedCall:output:04category_encoding_7/StatefulPartitionedCall:output:04category_encoding_8/StatefulPartitionedCall:output:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_101632
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identity?
NoOpNoOp*^category_encoding/StatefulPartitionedCall,^category_encoding_1/StatefulPartitionedCall,^category_encoding_2/StatefulPartitionedCall,^category_encoding_3/StatefulPartitionedCall,^category_encoding_4/StatefulPartitionedCall,^category_encoding_5/StatefulPartitionedCall,^category_encoding_6/StatefulPartitionedCall,^category_encoding_7/StatefulPartitionedCall,^category_encoding_8/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2.^string_lookup_1/None_Lookup/LookupTableFindV2.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2.^string_lookup_4/None_Lookup/LookupTableFindV2.^string_lookup_5/None_Lookup/LookupTableFindV2.^string_lookup_6/None_Lookup/LookupTableFindV2.^string_lookup_7/None_Lookup/LookupTableFindV2.^string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	2V
)category_encoding/StatefulPartitionedCall)category_encoding/StatefulPartitionedCall2Z
+category_encoding_1/StatefulPartitionedCall+category_encoding_1/StatefulPartitionedCall2Z
+category_encoding_2/StatefulPartitionedCall+category_encoding_2/StatefulPartitionedCall2Z
+category_encoding_3/StatefulPartitionedCall+category_encoding_3/StatefulPartitionedCall2Z
+category_encoding_4/StatefulPartitionedCall+category_encoding_4/StatefulPartitionedCall2Z
+category_encoding_5/StatefulPartitionedCall+category_encoding_5/StatefulPartitionedCall2Z
+category_encoding_6/StatefulPartitionedCall+category_encoding_6/StatefulPartitionedCall2Z
+category_encoding_7/StatefulPartitionedCall+category_encoding_7/StatefulPartitionedCall2Z
+category_encoding_8/StatefulPartitionedCall+category_encoding_8/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV22^
-string_lookup_4/None_Lookup/LookupTableFindV2-string_lookup_4/None_Lookup/LookupTableFindV22^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV22^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV22^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV22^
-string_lookup_8/None_Lookup/LookupTableFindV2-string_lookup_8/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?$
?
'__inference_model_1_layer_call_fn_12269
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18

unknown_19:	? @

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_achievementsinputs_appidinputs_average_playtimeinputs_categoriesinputs_developerinputs_englishinputs_genresinputs_median_playtimeinputs_nameinputs_negative_ratingsinputs_ownersinputs_platformsinputs_positive_ratingsinputs_priceinputs_publisherinputs_release_dateinputs_required_ageinputs_steamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*5
Tin.
,2*									*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
&'()*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_109472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
??
?
@__inference_model_layer_call_and_return_conditional_losses_12677
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tags>
:string_lookup_8_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_4_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_4_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
identity??category_encoding/Assert/Assert?!category_encoding_1/Assert/Assert?!category_encoding_2/Assert/Assert?!category_encoding_3/Assert/Assert?!category_encoding_4/Assert/Assert?!category_encoding_5/Assert/Assert?!category_encoding_6/Assert/Assert?!category_encoding_7/Assert/Assert?!category_encoding_8/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?-string_lookup_1/None_Lookup/LookupTableFindV2?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2?-string_lookup_4/None_Lookup/LookupTableFindV2?-string_lookup_5/None_Lookup/LookupTableFindV2?-string_lookup_6/None_Lookup/LookupTableFindV2?-string_lookup_7/None_Lookup/LookupTableFindV2?-string_lookup_8/None_Lookup/LookupTableFindV2?
-string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_8_none_lookup_lookuptablefindv2_table_handleinputs_owners;string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_8/None_Lookup/LookupTableFindV2?
string_lookup_8/IdentityIdentity6string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_8/Identity?
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handleinputs_steamspy_tags;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_7/None_Lookup/LookupTableFindV2?
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_7/Identity?
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handleinputs_genres;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_6/None_Lookup/LookupTableFindV2?
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_6/Identity?
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handleinputs_categories;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_5/None_Lookup/LookupTableFindV2?
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_5/Identity?
-string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_4_none_lookup_lookuptablefindv2_table_handleinputs_platforms;string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_4/None_Lookup/LookupTableFindV2?
string_lookup_4/IdentityIdentity6string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_4/Identity?
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handleinputs_publisher;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_3/None_Lookup/LookupTableFindV2?
string_lookup_3/IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_3/Identity?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_developer;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_2/Identity?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_release_date;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_name9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_appidinputs_englishinputs_required_ageinputs_achievementsinputs_positive_ratingsinputs_negative_ratingsinputs_average_playtimeinputs_median_playtimeinputs_price concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????	2
concatenate/concat?
normalization/subSubconcatenate/concat:output:0normalization_sub_y*
T0*'
_output_shapes
:?????????	2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:	2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMaxstring_lookup/Identity:output:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMinstring_lookup/Identity:output:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShapestring_lookup/Identity:output:0 ^category_encoding/Assert/Assert*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst ^category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const ^category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMaxstring_lookup/Identity:output:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
category_encoding_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_1/Const?
category_encoding_1/MaxMax!string_lookup_1/Identity:output:0"category_encoding_1/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_1/Max?
category_encoding_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_1/Const_1?
category_encoding_1/MinMin!string_lookup_1/Identity:output:0$category_encoding_1/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_1/Min{
category_encoding_1/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_1/Cast/x?
category_encoding_1/CastCast#category_encoding_1/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_1/Cast?
category_encoding_1/GreaterGreatercategory_encoding_1/Cast:y:0 category_encoding_1/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_1/Greater~
category_encoding_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_1/Cast_1/x?
category_encoding_1/Cast_1Cast%category_encoding_1/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_1/Cast_1?
 category_encoding_1/GreaterEqualGreaterEqual category_encoding_1/Min:output:0category_encoding_1/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/GreaterEqual?
category_encoding_1/LogicalAnd
LogicalAndcategory_encoding_1/Greater:z:0$category_encoding_1/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_1/LogicalAnd?
 category_encoding_1/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852"
 category_encoding_1/Assert/Const?
(category_encoding_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852*
(category_encoding_1/Assert/Assert/data_0?
!category_encoding_1/Assert/AssertAssert"category_encoding_1/LogicalAnd:z:01category_encoding_1/Assert/Assert/data_0:output:0 ^category_encoding/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_1/Assert/Assert?
"category_encoding_1/bincount/ShapeShape!string_lookup_1/Identity:output:0"^category_encoding_1/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst"^category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const"^category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMax!string_lookup_1/Identity:output:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
&category_encoding_1/bincount/maxlengthConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_1/bincount/maxlength?
$category_encoding_1/bincount/MinimumMinimum/category_encoding_1/bincount/maxlength:output:0(category_encoding_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Minimum?
$category_encoding_1/bincount/Const_2Const"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0(category_encoding_1/bincount/Minimum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
category_encoding_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_2/Const?
category_encoding_2/MaxMax!string_lookup_2/Identity:output:0"category_encoding_2/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_2/Max?
category_encoding_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_2/Const_1?
category_encoding_2/MinMin!string_lookup_2/Identity:output:0$category_encoding_2/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_2/Min{
category_encoding_2/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_2/Cast/x?
category_encoding_2/CastCast#category_encoding_2/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_2/Cast?
category_encoding_2/GreaterGreatercategory_encoding_2/Cast:y:0 category_encoding_2/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_2/Greater~
category_encoding_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_2/Cast_1/x?
category_encoding_2/Cast_1Cast%category_encoding_2/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_2/Cast_1?
 category_encoding_2/GreaterEqualGreaterEqual category_encoding_2/Min:output:0category_encoding_2/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/GreaterEqual?
category_encoding_2/LogicalAnd
LogicalAndcategory_encoding_2/Greater:z:0$category_encoding_2/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_2/LogicalAnd?
 category_encoding_2/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542"
 category_encoding_2/Assert/Const?
(category_encoding_2/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542*
(category_encoding_2/Assert/Assert/data_0?
!category_encoding_2/Assert/AssertAssert"category_encoding_2/LogicalAnd:z:01category_encoding_2/Assert/Assert/data_0:output:0"^category_encoding_1/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_2/Assert/Assert?
"category_encoding_2/bincount/ShapeShape!string_lookup_2/Identity:output:0"^category_encoding_2/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst"^category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const"^category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMax!string_lookup_2/Identity:output:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
&category_encoding_2/bincount/maxlengthConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_2/bincount/maxlength?
$category_encoding_2/bincount/MinimumMinimum/category_encoding_2/bincount/maxlength:output:0(category_encoding_2/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Minimum?
$category_encoding_2/bincount/Const_2Const"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincount!string_lookup_2/Identity:output:0(category_encoding_2/bincount/Minimum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMax!string_lookup_3/Identity:output:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMin!string_lookup_3/Identity:output:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0"^category_encoding_2/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShape!string_lookup_3/Identity:output:0"^category_encoding_3/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst"^category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const"^category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMax!string_lookup_3/Identity:output:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincount!string_lookup_3/Identity:output:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
category_encoding_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_4/Const?
category_encoding_4/MaxMax!string_lookup_4/Identity:output:0"category_encoding_4/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_4/Max?
category_encoding_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_4/Const_1?
category_encoding_4/MinMin!string_lookup_4/Identity:output:0$category_encoding_4/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_4/Minz
category_encoding_4/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
category_encoding_4/Cast/x?
category_encoding_4/CastCast#category_encoding_4/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_4/Cast?
category_encoding_4/GreaterGreatercategory_encoding_4/Cast:y:0 category_encoding_4/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_4/Greater~
category_encoding_4/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_4/Cast_1/x?
category_encoding_4/Cast_1Cast%category_encoding_4/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_4/Cast_1?
 category_encoding_4/GreaterEqualGreaterEqual category_encoding_4/Min:output:0category_encoding_4/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/GreaterEqual?
category_encoding_4/LogicalAnd
LogicalAndcategory_encoding_4/Greater:z:0$category_encoding_4/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_4/LogicalAnd?
 category_encoding_4/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52"
 category_encoding_4/Assert/Const?
(category_encoding_4/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52*
(category_encoding_4/Assert/Assert/data_0?
!category_encoding_4/Assert/AssertAssert"category_encoding_4/LogicalAnd:z:01category_encoding_4/Assert/Assert/data_0:output:0"^category_encoding_3/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_4/Assert/Assert?
"category_encoding_4/bincount/ShapeShape!string_lookup_4/Identity:output:0"^category_encoding_4/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst"^category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const"^category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMax!string_lookup_4/Identity:output:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
&category_encoding_4/bincount/maxlengthConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/maxlength?
$category_encoding_4/bincount/MinimumMinimum/category_encoding_4/bincount/maxlength:output:0(category_encoding_4/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Minimum?
$category_encoding_4/bincount/Const_2Const"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincount!string_lookup_4/Identity:output:0(category_encoding_4/bincount/Minimum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
category_encoding_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_5/Const?
category_encoding_5/MaxMax!string_lookup_5/Identity:output:0"category_encoding_5/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_5/Max?
category_encoding_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_5/Const_1?
category_encoding_5/MinMin!string_lookup_5/Identity:output:0$category_encoding_5/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_5/Min{
category_encoding_5/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_5/Cast/x?
category_encoding_5/CastCast#category_encoding_5/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_5/Cast?
category_encoding_5/GreaterGreatercategory_encoding_5/Cast:y:0 category_encoding_5/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_5/Greater~
category_encoding_5/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_5/Cast_1/x?
category_encoding_5/Cast_1Cast%category_encoding_5/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_5/Cast_1?
 category_encoding_5/GreaterEqualGreaterEqual category_encoding_5/Min:output:0category_encoding_5/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/GreaterEqual?
category_encoding_5/LogicalAnd
LogicalAndcategory_encoding_5/Greater:z:0$category_encoding_5/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_5/LogicalAnd?
 category_encoding_5/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952"
 category_encoding_5/Assert/Const?
(category_encoding_5/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952*
(category_encoding_5/Assert/Assert/data_0?
!category_encoding_5/Assert/AssertAssert"category_encoding_5/LogicalAnd:z:01category_encoding_5/Assert/Assert/data_0:output:0"^category_encoding_4/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_5/Assert/Assert?
"category_encoding_5/bincount/ShapeShape!string_lookup_5/Identity:output:0"^category_encoding_5/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst"^category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const"^category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMax!string_lookup_5/Identity:output:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
&category_encoding_5/bincount/maxlengthConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_5/bincount/maxlength?
$category_encoding_5/bincount/MinimumMinimum/category_encoding_5/bincount/maxlength:output:0(category_encoding_5/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Minimum?
$category_encoding_5/bincount/Const_2Const"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincount!string_lookup_5/Identity:output:0(category_encoding_5/bincount/Minimum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
category_encoding_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_6/Const?
category_encoding_6/MaxMax!string_lookup_6/Identity:output:0"category_encoding_6/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_6/Max?
category_encoding_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_6/Const_1?
category_encoding_6/MinMin!string_lookup_6/Identity:output:0$category_encoding_6/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_6/Min{
category_encoding_6/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_6/Cast/x?
category_encoding_6/CastCast#category_encoding_6/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_6/Cast?
category_encoding_6/GreaterGreatercategory_encoding_6/Cast:y:0 category_encoding_6/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_6/Greater~
category_encoding_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_6/Cast_1/x?
category_encoding_6/Cast_1Cast%category_encoding_6/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_6/Cast_1?
 category_encoding_6/GreaterEqualGreaterEqual category_encoding_6/Min:output:0category_encoding_6/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/GreaterEqual?
category_encoding_6/LogicalAnd
LogicalAndcategory_encoding_6/Greater:z:0$category_encoding_6/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_6/LogicalAnd?
 category_encoding_6/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172"
 category_encoding_6/Assert/Const?
(category_encoding_6/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172*
(category_encoding_6/Assert/Assert/data_0?
!category_encoding_6/Assert/AssertAssert"category_encoding_6/LogicalAnd:z:01category_encoding_6/Assert/Assert/data_0:output:0"^category_encoding_5/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_6/Assert/Assert?
"category_encoding_6/bincount/ShapeShape!string_lookup_6/Identity:output:0"^category_encoding_6/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst"^category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const"^category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMax!string_lookup_6/Identity:output:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
&category_encoding_6/bincount/maxlengthConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_6/bincount/maxlength?
$category_encoding_6/bincount/MinimumMinimum/category_encoding_6/bincount/maxlength:output:0(category_encoding_6/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Minimum?
$category_encoding_6/bincount/Const_2Const"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincount!string_lookup_6/Identity:output:0(category_encoding_6/bincount/Minimum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
category_encoding_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_7/Const?
category_encoding_7/MaxMax!string_lookup_7/Identity:output:0"category_encoding_7/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_7/Max?
category_encoding_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_7/Const_1?
category_encoding_7/MinMin!string_lookup_7/Identity:output:0$category_encoding_7/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_7/Min{
category_encoding_7/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_7/Cast/x?
category_encoding_7/CastCast#category_encoding_7/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_7/Cast?
category_encoding_7/GreaterGreatercategory_encoding_7/Cast:y:0 category_encoding_7/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_7/Greater~
category_encoding_7/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_7/Cast_1/x?
category_encoding_7/Cast_1Cast%category_encoding_7/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_7/Cast_1?
 category_encoding_7/GreaterEqualGreaterEqual category_encoding_7/Min:output:0category_encoding_7/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/GreaterEqual?
category_encoding_7/LogicalAnd
LogicalAndcategory_encoding_7/Greater:z:0$category_encoding_7/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_7/LogicalAnd?
 category_encoding_7/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442"
 category_encoding_7/Assert/Const?
(category_encoding_7/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442*
(category_encoding_7/Assert/Assert/data_0?
!category_encoding_7/Assert/AssertAssert"category_encoding_7/LogicalAnd:z:01category_encoding_7/Assert/Assert/data_0:output:0"^category_encoding_6/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_7/Assert/Assert?
"category_encoding_7/bincount/ShapeShape!string_lookup_7/Identity:output:0"^category_encoding_7/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst"^category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const"^category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMax!string_lookup_7/Identity:output:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
&category_encoding_7/bincount/maxlengthConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_7/bincount/maxlength?
$category_encoding_7/bincount/MinimumMinimum/category_encoding_7/bincount/maxlength:output:0(category_encoding_7/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Minimum?
$category_encoding_7/bincount/Const_2Const"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincount!string_lookup_7/Identity:output:0(category_encoding_7/bincount/Minimum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_7/bincount/DenseBincount?
category_encoding_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_8/Const?
category_encoding_8/MaxMax!string_lookup_8/Identity:output:0"category_encoding_8/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_8/Max?
category_encoding_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_8/Const_1?
category_encoding_8/MinMin!string_lookup_8/Identity:output:0$category_encoding_8/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_8/Minz
category_encoding_8/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
category_encoding_8/Cast/x?
category_encoding_8/CastCast#category_encoding_8/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_8/Cast?
category_encoding_8/GreaterGreatercategory_encoding_8/Cast:y:0 category_encoding_8/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_8/Greater~
category_encoding_8/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_8/Cast_1/x?
category_encoding_8/Cast_1Cast%category_encoding_8/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_8/Cast_1?
 category_encoding_8/GreaterEqualGreaterEqual category_encoding_8/Min:output:0category_encoding_8/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/GreaterEqual?
category_encoding_8/LogicalAnd
LogicalAndcategory_encoding_8/Greater:z:0$category_encoding_8/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_8/LogicalAnd?
 category_encoding_8/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112"
 category_encoding_8/Assert/Const?
(category_encoding_8/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112*
(category_encoding_8/Assert/Assert/data_0?
!category_encoding_8/Assert/AssertAssert"category_encoding_8/LogicalAnd:z:01category_encoding_8/Assert/Assert/data_0:output:0"^category_encoding_7/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_8/Assert/Assert?
"category_encoding_8/bincount/ShapeShape!string_lookup_8/Identity:output:0"^category_encoding_8/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_8/bincount/Shape?
"category_encoding_8/bincount/ConstConst"^category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_8/bincount/Const?
!category_encoding_8/bincount/ProdProd+category_encoding_8/bincount/Shape:output:0+category_encoding_8/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_8/bincount/Prod?
&category_encoding_8/bincount/Greater/yConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_8/bincount/Greater/y?
$category_encoding_8/bincount/GreaterGreater*category_encoding_8/bincount/Prod:output:0/category_encoding_8/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_8/bincount/Greater?
!category_encoding_8/bincount/CastCast(category_encoding_8/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_8/bincount/Cast?
$category_encoding_8/bincount/Const_1Const"^category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_8/bincount/Const_1?
 category_encoding_8/bincount/MaxMax!string_lookup_8/Identity:output:0-category_encoding_8/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/bincount/Max?
"category_encoding_8/bincount/add/yConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_8/bincount/add/y?
 category_encoding_8/bincount/addAddV2)category_encoding_8/bincount/Max:output:0+category_encoding_8/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/bincount/add?
 category_encoding_8/bincount/mulMul%category_encoding_8/bincount/Cast:y:0$category_encoding_8/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/bincount/mul?
&category_encoding_8/bincount/minlengthConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_8/bincount/minlength?
$category_encoding_8/bincount/MaximumMaximum/category_encoding_8/bincount/minlength:output:0$category_encoding_8/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_8/bincount/Maximum?
&category_encoding_8/bincount/maxlengthConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_8/bincount/maxlength?
$category_encoding_8/bincount/MinimumMinimum/category_encoding_8/bincount/maxlength:output:0(category_encoding_8/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_8/bincount/Minimum?
$category_encoding_8/bincount/Const_2Const"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_8/bincount/Const_2?
*category_encoding_8/bincount/DenseBincountDenseBincount!string_lookup_8/Identity:output:0(category_encoding_8/bincount/Minimum:z:0-category_encoding_8/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_8/bincount/DenseBincountx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2normalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:03category_encoding_8/bincount/DenseBincount:output:0"concatenate_1/concat/axis:output:0*
N
*
T0*(
_output_shapes
:?????????? 2
concatenate_1/concaty
IdentityIdentityconcatenate_1/concat:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identity?
NoOpNoOp ^category_encoding/Assert/Assert"^category_encoding_1/Assert/Assert"^category_encoding_2/Assert/Assert"^category_encoding_3/Assert/Assert"^category_encoding_4/Assert/Assert"^category_encoding_5/Assert/Assert"^category_encoding_6/Assert/Assert"^category_encoding_7/Assert/Assert"^category_encoding_8/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2.^string_lookup_1/None_Lookup/LookupTableFindV2.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2.^string_lookup_4/None_Lookup/LookupTableFindV2.^string_lookup_5/None_Lookup/LookupTableFindV2.^string_lookup_6/None_Lookup/LookupTableFindV2.^string_lookup_7/None_Lookup/LookupTableFindV2.^string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2F
!category_encoding_1/Assert/Assert!category_encoding_1/Assert/Assert2F
!category_encoding_2/Assert/Assert!category_encoding_2/Assert/Assert2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2F
!category_encoding_4/Assert/Assert!category_encoding_4/Assert/Assert2F
!category_encoding_5/Assert/Assert!category_encoding_5/Assert/Assert2F
!category_encoding_6/Assert/Assert!category_encoding_6/Assert/Assert2F
!category_encoding_7/Assert/Assert!category_encoding_7/Assert/Assert2F
!category_encoding_8/Assert/Assert!category_encoding_8/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV22^
-string_lookup_4/None_Lookup/LookupTableFindV2-string_lookup_4/None_Lookup/LookupTableFindV22^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV22^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV22^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV22^
-string_lookup_8/None_Lookup/LookupTableFindV2-string_lookup_8/None_Lookup/LookupTableFindV2:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
?
__inference__initializer_137197
3key_value_init6522_lookuptableimportv2_table_handle/
+key_value_init6522_lookuptableimportv2_keys1
-key_value_init6522_lookuptableimportv2_values	
identity??&key_value_init6522/LookupTableImportV2?
&key_value_init6522/LookupTableImportV2LookupTableImportV23key_value_init6522_lookuptableimportv2_table_handle+key_value_init6522_lookuptableimportv2_keys-key_value_init6522_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6522/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6522/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6522/LookupTableImportV2&key_value_init6522/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
? 
}
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_13538

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
}
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_13499

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_13015
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tags>
:string_lookup_8_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_8_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_7_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_7_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_6_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_6_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_5_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_5_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_4_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_4_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_3_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_3_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	
normalization_sub_y
normalization_sqrt_x
identity??category_encoding/Assert/Assert?!category_encoding_1/Assert/Assert?!category_encoding_2/Assert/Assert?!category_encoding_3/Assert/Assert?!category_encoding_4/Assert/Assert?!category_encoding_5/Assert/Assert?!category_encoding_6/Assert/Assert?!category_encoding_7/Assert/Assert?!category_encoding_8/Assert/Assert?+string_lookup/None_Lookup/LookupTableFindV2?-string_lookup_1/None_Lookup/LookupTableFindV2?-string_lookup_2/None_Lookup/LookupTableFindV2?-string_lookup_3/None_Lookup/LookupTableFindV2?-string_lookup_4/None_Lookup/LookupTableFindV2?-string_lookup_5/None_Lookup/LookupTableFindV2?-string_lookup_6/None_Lookup/LookupTableFindV2?-string_lookup_7/None_Lookup/LookupTableFindV2?-string_lookup_8/None_Lookup/LookupTableFindV2?
-string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_8_none_lookup_lookuptablefindv2_table_handleinputs_owners;string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_8/None_Lookup/LookupTableFindV2?
string_lookup_8/IdentityIdentity6string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_8/Identity?
-string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_7_none_lookup_lookuptablefindv2_table_handleinputs_steamspy_tags;string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_7/None_Lookup/LookupTableFindV2?
string_lookup_7/IdentityIdentity6string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_7/Identity?
-string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_6_none_lookup_lookuptablefindv2_table_handleinputs_genres;string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_6/None_Lookup/LookupTableFindV2?
string_lookup_6/IdentityIdentity6string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_6/Identity?
-string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_5_none_lookup_lookuptablefindv2_table_handleinputs_categories;string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_5/None_Lookup/LookupTableFindV2?
string_lookup_5/IdentityIdentity6string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_5/Identity?
-string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_4_none_lookup_lookuptablefindv2_table_handleinputs_platforms;string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_4/None_Lookup/LookupTableFindV2?
string_lookup_4/IdentityIdentity6string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_4/Identity?
-string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_3_none_lookup_lookuptablefindv2_table_handleinputs_publisher;string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_3/None_Lookup/LookupTableFindV2?
string_lookup_3/IdentityIdentity6string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_3/Identity?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs_developer;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_2/Identity?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs_release_date;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup_1/Identity?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs_name9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2
string_lookup/Identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2inputs_appidinputs_englishinputs_required_ageinputs_achievementsinputs_positive_ratingsinputs_negative_ratingsinputs_average_playtimeinputs_median_playtimeinputs_price concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????	2
concatenate/concat?
normalization/subSubconcatenate/concat:output:0normalization_sub_y*
T0*'
_output_shapes
:?????????	2
normalization/subo
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:	2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2
normalization/truediv?
category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const?
category_encoding/MaxMaxstring_lookup/Identity:output:0 category_encoding/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding/Max?
category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding/Const_1?
category_encoding/MinMinstring_lookup/Identity:output:0"category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding/Minw
category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding/Cast/x?
category_encoding/CastCast!category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast?
category_encoding/GreaterGreatercategory_encoding/Cast:y:0category_encoding/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding/Greaterz
category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding/Cast_1/x?
category_encoding/Cast_1Cast#category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding/Cast_1?
category_encoding/GreaterEqualGreaterEqualcategory_encoding/Min:output:0category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2 
category_encoding/GreaterEqual?
category_encoding/LogicalAnd
LogicalAndcategory_encoding/Greater:z:0"category_encoding/GreaterEqual:z:0*
_output_shapes
: 2
category_encoding/LogicalAnd?
category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502 
category_encoding/Assert/Const?
&category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502(
&category_encoding/Assert/Assert/data_0?
category_encoding/Assert/AssertAssert category_encoding/LogicalAnd:z:0/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2!
category_encoding/Assert/Assert?
 category_encoding/bincount/ShapeShapestring_lookup/Identity:output:0 ^category_encoding/Assert/Assert*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst ^category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const ^category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMaxstring_lookup/Identity:output:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
$category_encoding/bincount/maxlengthConst ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2&
$category_encoding/bincount/maxlength?
"category_encoding/bincount/MinimumMinimum-category_encoding/bincount/maxlength:output:0&category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Minimum?
"category_encoding/bincount/Const_2Const ^category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincountstring_lookup/Identity:output:0&category_encoding/bincount/Minimum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
category_encoding_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_1/Const?
category_encoding_1/MaxMax!string_lookup_1/Identity:output:0"category_encoding_1/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_1/Max?
category_encoding_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_1/Const_1?
category_encoding_1/MinMin!string_lookup_1/Identity:output:0$category_encoding_1/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_1/Min{
category_encoding_1/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_1/Cast/x?
category_encoding_1/CastCast#category_encoding_1/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_1/Cast?
category_encoding_1/GreaterGreatercategory_encoding_1/Cast:y:0 category_encoding_1/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_1/Greater~
category_encoding_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_1/Cast_1/x?
category_encoding_1/Cast_1Cast%category_encoding_1/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_1/Cast_1?
 category_encoding_1/GreaterEqualGreaterEqual category_encoding_1/Min:output:0category_encoding_1/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/GreaterEqual?
category_encoding_1/LogicalAnd
LogicalAndcategory_encoding_1/Greater:z:0$category_encoding_1/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_1/LogicalAnd?
 category_encoding_1/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852"
 category_encoding_1/Assert/Const?
(category_encoding_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6852*
(category_encoding_1/Assert/Assert/data_0?
!category_encoding_1/Assert/AssertAssert"category_encoding_1/LogicalAnd:z:01category_encoding_1/Assert/Assert/data_0:output:0 ^category_encoding/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_1/Assert/Assert?
"category_encoding_1/bincount/ShapeShape!string_lookup_1/Identity:output:0"^category_encoding_1/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst"^category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const"^category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMax!string_lookup_1/Identity:output:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
&category_encoding_1/bincount/maxlengthConst"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_1/bincount/maxlength?
$category_encoding_1/bincount/MinimumMinimum/category_encoding_1/bincount/maxlength:output:0(category_encoding_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Minimum?
$category_encoding_1/bincount/Const_2Const"^category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincount!string_lookup_1/Identity:output:0(category_encoding_1/bincount/Minimum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
category_encoding_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_2/Const?
category_encoding_2/MaxMax!string_lookup_2/Identity:output:0"category_encoding_2/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_2/Max?
category_encoding_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_2/Const_1?
category_encoding_2/MinMin!string_lookup_2/Identity:output:0$category_encoding_2/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_2/Min{
category_encoding_2/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_2/Cast/x?
category_encoding_2/CastCast#category_encoding_2/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_2/Cast?
category_encoding_2/GreaterGreatercategory_encoding_2/Cast:y:0 category_encoding_2/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_2/Greater~
category_encoding_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_2/Cast_1/x?
category_encoding_2/Cast_1Cast%category_encoding_2/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_2/Cast_1?
 category_encoding_2/GreaterEqualGreaterEqual category_encoding_2/Min:output:0category_encoding_2/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/GreaterEqual?
category_encoding_2/LogicalAnd
LogicalAndcategory_encoding_2/Greater:z:0$category_encoding_2/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_2/LogicalAnd?
 category_encoding_2/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542"
 category_encoding_2/Assert/Const?
(category_encoding_2/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=6542*
(category_encoding_2/Assert/Assert/data_0?
!category_encoding_2/Assert/AssertAssert"category_encoding_2/LogicalAnd:z:01category_encoding_2/Assert/Assert/data_0:output:0"^category_encoding_1/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_2/Assert/Assert?
"category_encoding_2/bincount/ShapeShape!string_lookup_2/Identity:output:0"^category_encoding_2/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst"^category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const"^category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMax!string_lookup_2/Identity:output:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
&category_encoding_2/bincount/maxlengthConst"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_2/bincount/maxlength?
$category_encoding_2/bincount/MinimumMinimum/category_encoding_2/bincount/maxlength:output:0(category_encoding_2/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Minimum?
$category_encoding_2/bincount/Const_2Const"^category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincount!string_lookup_2/Identity:output:0(category_encoding_2/bincount/Minimum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const?
category_encoding_3/MaxMax!string_lookup_3/Identity:output:0"category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Max?
category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_3/Const_1?
category_encoding_3/MinMin!string_lookup_3/Identity:output:0$category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Min{
category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_3/Cast/x?
category_encoding_3/CastCast#category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast?
category_encoding_3/GreaterGreatercategory_encoding_3/Cast:y:0 category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_3/Greater~
category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_3/Cast_1/x?
category_encoding_3/Cast_1Cast%category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_3/Cast_1?
 category_encoding_3/GreaterEqualGreaterEqual category_encoding_3/Min:output:0category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/GreaterEqual?
category_encoding_3/LogicalAnd
LogicalAndcategory_encoding_3/Greater:z:0$category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_3/LogicalAnd?
 category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792"
 category_encoding_3/Assert/Const?
(category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4792*
(category_encoding_3/Assert/Assert/data_0?
!category_encoding_3/Assert/AssertAssert"category_encoding_3/LogicalAnd:z:01category_encoding_3/Assert/Assert/data_0:output:0"^category_encoding_2/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_3/Assert/Assert?
"category_encoding_3/bincount/ShapeShape!string_lookup_3/Identity:output:0"^category_encoding_3/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst"^category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const"^category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMax!string_lookup_3/Identity:output:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
&category_encoding_3/bincount/maxlengthConst"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_3/bincount/maxlength?
$category_encoding_3/bincount/MinimumMinimum/category_encoding_3/bincount/maxlength:output:0(category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Minimum?
$category_encoding_3/bincount/Const_2Const"^category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincount!string_lookup_3/Identity:output:0(category_encoding_3/bincount/Minimum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
category_encoding_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_4/Const?
category_encoding_4/MaxMax!string_lookup_4/Identity:output:0"category_encoding_4/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_4/Max?
category_encoding_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_4/Const_1?
category_encoding_4/MinMin!string_lookup_4/Identity:output:0$category_encoding_4/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_4/Minz
category_encoding_4/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
category_encoding_4/Cast/x?
category_encoding_4/CastCast#category_encoding_4/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_4/Cast?
category_encoding_4/GreaterGreatercategory_encoding_4/Cast:y:0 category_encoding_4/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_4/Greater~
category_encoding_4/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_4/Cast_1/x?
category_encoding_4/Cast_1Cast%category_encoding_4/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_4/Cast_1?
 category_encoding_4/GreaterEqualGreaterEqual category_encoding_4/Min:output:0category_encoding_4/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/GreaterEqual?
category_encoding_4/LogicalAnd
LogicalAndcategory_encoding_4/Greater:z:0$category_encoding_4/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_4/LogicalAnd?
 category_encoding_4/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52"
 category_encoding_4/Assert/Const?
(category_encoding_4/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=52*
(category_encoding_4/Assert/Assert/data_0?
!category_encoding_4/Assert/AssertAssert"category_encoding_4/LogicalAnd:z:01category_encoding_4/Assert/Assert/data_0:output:0"^category_encoding_3/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_4/Assert/Assert?
"category_encoding_4/bincount/ShapeShape!string_lookup_4/Identity:output:0"^category_encoding_4/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst"^category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const"^category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMax!string_lookup_4/Identity:output:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
&category_encoding_4/bincount/maxlengthConst"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/maxlength?
$category_encoding_4/bincount/MinimumMinimum/category_encoding_4/bincount/maxlength:output:0(category_encoding_4/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Minimum?
$category_encoding_4/bincount/Const_2Const"^category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincount!string_lookup_4/Identity:output:0(category_encoding_4/bincount/Minimum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
category_encoding_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_5/Const?
category_encoding_5/MaxMax!string_lookup_5/Identity:output:0"category_encoding_5/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_5/Max?
category_encoding_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_5/Const_1?
category_encoding_5/MinMin!string_lookup_5/Identity:output:0$category_encoding_5/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_5/Min{
category_encoding_5/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_5/Cast/x?
category_encoding_5/CastCast#category_encoding_5/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_5/Cast?
category_encoding_5/GreaterGreatercategory_encoding_5/Cast:y:0 category_encoding_5/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_5/Greater~
category_encoding_5/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_5/Cast_1/x?
category_encoding_5/Cast_1Cast%category_encoding_5/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_5/Cast_1?
 category_encoding_5/GreaterEqualGreaterEqual category_encoding_5/Min:output:0category_encoding_5/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/GreaterEqual?
category_encoding_5/LogicalAnd
LogicalAndcategory_encoding_5/Greater:z:0$category_encoding_5/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_5/LogicalAnd?
 category_encoding_5/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952"
 category_encoding_5/Assert/Const?
(category_encoding_5/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=4952*
(category_encoding_5/Assert/Assert/data_0?
!category_encoding_5/Assert/AssertAssert"category_encoding_5/LogicalAnd:z:01category_encoding_5/Assert/Assert/data_0:output:0"^category_encoding_4/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_5/Assert/Assert?
"category_encoding_5/bincount/ShapeShape!string_lookup_5/Identity:output:0"^category_encoding_5/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst"^category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const"^category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMax!string_lookup_5/Identity:output:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
&category_encoding_5/bincount/maxlengthConst"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_5/bincount/maxlength?
$category_encoding_5/bincount/MinimumMinimum/category_encoding_5/bincount/maxlength:output:0(category_encoding_5/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Minimum?
$category_encoding_5/bincount/Const_2Const"^category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincount!string_lookup_5/Identity:output:0(category_encoding_5/bincount/Minimum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
category_encoding_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_6/Const?
category_encoding_6/MaxMax!string_lookup_6/Identity:output:0"category_encoding_6/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_6/Max?
category_encoding_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_6/Const_1?
category_encoding_6/MinMin!string_lookup_6/Identity:output:0$category_encoding_6/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_6/Min{
category_encoding_6/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_6/Cast/x?
category_encoding_6/CastCast#category_encoding_6/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_6/Cast?
category_encoding_6/GreaterGreatercategory_encoding_6/Cast:y:0 category_encoding_6/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_6/Greater~
category_encoding_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_6/Cast_1/x?
category_encoding_6/Cast_1Cast%category_encoding_6/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_6/Cast_1?
 category_encoding_6/GreaterEqualGreaterEqual category_encoding_6/Min:output:0category_encoding_6/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/GreaterEqual?
category_encoding_6/LogicalAnd
LogicalAndcategory_encoding_6/Greater:z:0$category_encoding_6/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_6/LogicalAnd?
 category_encoding_6/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172"
 category_encoding_6/Assert/Const?
(category_encoding_6/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=2172*
(category_encoding_6/Assert/Assert/data_0?
!category_encoding_6/Assert/AssertAssert"category_encoding_6/LogicalAnd:z:01category_encoding_6/Assert/Assert/data_0:output:0"^category_encoding_5/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_6/Assert/Assert?
"category_encoding_6/bincount/ShapeShape!string_lookup_6/Identity:output:0"^category_encoding_6/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst"^category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const"^category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMax!string_lookup_6/Identity:output:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
&category_encoding_6/bincount/maxlengthConst"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_6/bincount/maxlength?
$category_encoding_6/bincount/MinimumMinimum/category_encoding_6/bincount/maxlength:output:0(category_encoding_6/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Minimum?
$category_encoding_6/bincount/Const_2Const"^category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincount!string_lookup_6/Identity:output:0(category_encoding_6/bincount/Minimum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
category_encoding_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_7/Const?
category_encoding_7/MaxMax!string_lookup_7/Identity:output:0"category_encoding_7/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_7/Max?
category_encoding_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_7/Const_1?
category_encoding_7/MinMin!string_lookup_7/Identity:output:0$category_encoding_7/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_7/Min{
category_encoding_7/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
category_encoding_7/Cast/x?
category_encoding_7/CastCast#category_encoding_7/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_7/Cast?
category_encoding_7/GreaterGreatercategory_encoding_7/Cast:y:0 category_encoding_7/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_7/Greater~
category_encoding_7/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_7/Cast_1/x?
category_encoding_7/Cast_1Cast%category_encoding_7/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_7/Cast_1?
 category_encoding_7/GreaterEqualGreaterEqual category_encoding_7/Min:output:0category_encoding_7/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/GreaterEqual?
category_encoding_7/LogicalAnd
LogicalAndcategory_encoding_7/Greater:z:0$category_encoding_7/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_7/LogicalAnd?
 category_encoding_7/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442"
 category_encoding_7/Assert/Const?
(category_encoding_7/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442*
(category_encoding_7/Assert/Assert/data_0?
!category_encoding_7/Assert/AssertAssert"category_encoding_7/LogicalAnd:z:01category_encoding_7/Assert/Assert/data_0:output:0"^category_encoding_6/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_7/Assert/Assert?
"category_encoding_7/bincount/ShapeShape!string_lookup_7/Identity:output:0"^category_encoding_7/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst"^category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const"^category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMax!string_lookup_7/Identity:output:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
&category_encoding_7/bincount/maxlengthConst"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2(
&category_encoding_7/bincount/maxlength?
$category_encoding_7/bincount/MinimumMinimum/category_encoding_7/bincount/maxlength:output:0(category_encoding_7/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Minimum?
$category_encoding_7/bincount/Const_2Const"^category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincount!string_lookup_7/Identity:output:0(category_encoding_7/bincount/Minimum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2,
*category_encoding_7/bincount/DenseBincount?
category_encoding_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_8/Const?
category_encoding_8/MaxMax!string_lookup_8/Identity:output:0"category_encoding_8/Const:output:0*
T0	*
_output_shapes
: 2
category_encoding_8/Max?
category_encoding_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
category_encoding_8/Const_1?
category_encoding_8/MinMin!string_lookup_8/Identity:output:0$category_encoding_8/Const_1:output:0*
T0	*
_output_shapes
: 2
category_encoding_8/Minz
category_encoding_8/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
category_encoding_8/Cast/x?
category_encoding_8/CastCast#category_encoding_8/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_8/Cast?
category_encoding_8/GreaterGreatercategory_encoding_8/Cast:y:0 category_encoding_8/Max:output:0*
T0	*
_output_shapes
: 2
category_encoding_8/Greater~
category_encoding_8/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2
category_encoding_8/Cast_1/x?
category_encoding_8/Cast_1Cast%category_encoding_8/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
category_encoding_8/Cast_1?
 category_encoding_8/GreaterEqualGreaterEqual category_encoding_8/Min:output:0category_encoding_8/Cast_1:y:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/GreaterEqual?
category_encoding_8/LogicalAnd
LogicalAndcategory_encoding_8/Greater:z:0$category_encoding_8/GreaterEqual:z:0*
_output_shapes
: 2 
category_encoding_8/LogicalAnd?
 category_encoding_8/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112"
 category_encoding_8/Assert/Const?
(category_encoding_8/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=112*
(category_encoding_8/Assert/Assert/data_0?
!category_encoding_8/Assert/AssertAssert"category_encoding_8/LogicalAnd:z:01category_encoding_8/Assert/Assert/data_0:output:0"^category_encoding_7/Assert/Assert*

T
2*
_output_shapes
 2#
!category_encoding_8/Assert/Assert?
"category_encoding_8/bincount/ShapeShape!string_lookup_8/Identity:output:0"^category_encoding_8/Assert/Assert*
T0	*
_output_shapes
:2$
"category_encoding_8/bincount/Shape?
"category_encoding_8/bincount/ConstConst"^category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_8/bincount/Const?
!category_encoding_8/bincount/ProdProd+category_encoding_8/bincount/Shape:output:0+category_encoding_8/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_8/bincount/Prod?
&category_encoding_8/bincount/Greater/yConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_8/bincount/Greater/y?
$category_encoding_8/bincount/GreaterGreater*category_encoding_8/bincount/Prod:output:0/category_encoding_8/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_8/bincount/Greater?
!category_encoding_8/bincount/CastCast(category_encoding_8/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_8/bincount/Cast?
$category_encoding_8/bincount/Const_1Const"^category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_8/bincount/Const_1?
 category_encoding_8/bincount/MaxMax!string_lookup_8/Identity:output:0-category_encoding_8/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/bincount/Max?
"category_encoding_8/bincount/add/yConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_8/bincount/add/y?
 category_encoding_8/bincount/addAddV2)category_encoding_8/bincount/Max:output:0+category_encoding_8/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/bincount/add?
 category_encoding_8/bincount/mulMul%category_encoding_8/bincount/Cast:y:0$category_encoding_8/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_8/bincount/mul?
&category_encoding_8/bincount/minlengthConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_8/bincount/minlength?
$category_encoding_8/bincount/MaximumMaximum/category_encoding_8/bincount/minlength:output:0$category_encoding_8/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_8/bincount/Maximum?
&category_encoding_8/bincount/maxlengthConst"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_8/bincount/maxlength?
$category_encoding_8/bincount/MinimumMinimum/category_encoding_8/bincount/maxlength:output:0(category_encoding_8/bincount/Maximum:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_8/bincount/Minimum?
$category_encoding_8/bincount/Const_2Const"^category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_8/bincount/Const_2?
*category_encoding_8/bincount/DenseBincountDenseBincount!string_lookup_8/Identity:output:0(category_encoding_8/bincount/Minimum:z:0-category_encoding_8/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_8/bincount/DenseBincountx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2normalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:03category_encoding_8/bincount/DenseBincount:output:0"concatenate_1/concat/axis:output:0*
N
*
T0*(
_output_shapes
:?????????? 2
concatenate_1/concaty
IdentityIdentityconcatenate_1/concat:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identity?
NoOpNoOp ^category_encoding/Assert/Assert"^category_encoding_1/Assert/Assert"^category_encoding_2/Assert/Assert"^category_encoding_3/Assert/Assert"^category_encoding_4/Assert/Assert"^category_encoding_5/Assert/Assert"^category_encoding_6/Assert/Assert"^category_encoding_7/Assert/Assert"^category_encoding_8/Assert/Assert,^string_lookup/None_Lookup/LookupTableFindV2.^string_lookup_1/None_Lookup/LookupTableFindV2.^string_lookup_2/None_Lookup/LookupTableFindV2.^string_lookup_3/None_Lookup/LookupTableFindV2.^string_lookup_4/None_Lookup/LookupTableFindV2.^string_lookup_5/None_Lookup/LookupTableFindV2.^string_lookup_6/None_Lookup/LookupTableFindV2.^string_lookup_7/None_Lookup/LookupTableFindV2.^string_lookup_8/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	2B
category_encoding/Assert/Assertcategory_encoding/Assert/Assert2F
!category_encoding_1/Assert/Assert!category_encoding_1/Assert/Assert2F
!category_encoding_2/Assert/Assert!category_encoding_2/Assert/Assert2F
!category_encoding_3/Assert/Assert!category_encoding_3/Assert/Assert2F
!category_encoding_4/Assert/Assert!category_encoding_4/Assert/Assert2F
!category_encoding_5/Assert/Assert!category_encoding_5/Assert/Assert2F
!category_encoding_6/Assert/Assert!category_encoding_6/Assert/Assert2F
!category_encoding_7/Assert/Assert!category_encoding_7/Assert/Assert2F
!category_encoding_8/Assert/Assert!category_encoding_8/Assert/Assert2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV22^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV22^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV22^
-string_lookup_3/None_Lookup/LookupTableFindV2-string_lookup_3/None_Lookup/LookupTableFindV22^
-string_lookup_4/None_Lookup/LookupTableFindV2-string_lookup_4/None_Lookup/LookupTableFindV22^
-string_lookup_5/None_Lookup/LookupTableFindV2-string_lookup_5/None_Lookup/LookupTableFindV22^
-string_lookup_6/None_Lookup/LookupTableFindV2-string_lookup_6/None_Lookup/LookupTableFindV22^
-string_lookup_7/None_Lookup/LookupTableFindV2-string_lookup_7/None_Lookup/LookupTableFindV22^
-string_lookup_8/None_Lookup/LookupTableFindV2-string_lookup_8/None_Lookup/LookupTableFindV2:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_13636
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:?????????? 2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:??????????:??????????:??????????:??????????:?????????:??????????:??????????:??????????:?????????:Q M
'
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/2:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/3:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/6:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/7:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/9
?
?
__inference_<lambda>_138827
3key_value_init6626_lookuptableimportv2_table_handle/
+key_value_init6626_lookuptableimportv2_keys1
-key_value_init6626_lookuptableimportv2_values	
identity??&key_value_init6626/LookupTableImportV2?
&key_value_init6626/LookupTableImportV2LookupTableImportV23key_value_init6626_lookuptableimportv2_table_handle+key_value_init6626_lookuptableimportv2_keys-key_value_init6626_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6626/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6626/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6626/LookupTableImportV2&key_value_init6626/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
? 
}
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_10110

inputs	
identity??Assert/Assert_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstJ
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: 2
Maxc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1L
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: 2
MinS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastV
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: 2	
GreaterV
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2

Cast_1/x[
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast_1g
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: 2
GreaterEqual]

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: 2

LogicalAnd?
Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442
Assert/Const?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=7442
Assert/Assert/data_0y
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assertf
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:2
bincount/Shapez
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 2
bincount/Consty
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: 2
bincount/Prodz
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 2
bincount/Greater/y?
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: 2
bincount/Greaterl
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2
bincount/Cast?
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       2
bincount/Const_1g
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: 2
bincount/Maxr
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R2
bincount/add/yv
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: 2
bincount/addi
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: 2
bincount/mul{
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/minlength
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: 2
bincount/Maximum{
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?2
bincount/maxlength?
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: 2
bincount/Minimumw
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB 2
bincount/Const_2?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2
bincount/DenseBincount{
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity^
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_10163

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:?????????? 2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:??????????:??????????:??????????:??????????:?????????:??????????:??????????:??????????:?????????:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_13660

inputs1
matmul_readvariableop_resource:	? @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
__inference__initializer_137017
3key_value_init6470_lookuptableimportv2_table_handle/
+key_value_init6470_lookuptableimportv2_keys1
-key_value_init6470_lookuptableimportv2_values	
identity??&key_value_init6470/LookupTableImportV2?
&key_value_init6470/LookupTableImportV2LookupTableImportV23key_value_init6470_lookuptableimportv2_table_handle+key_value_init6470_lookuptableimportv2_keys-key_value_init6470_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6470/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6470/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6470/LookupTableImportV2&key_value_init6470/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
__inference__initializer_137917
3key_value_init6730_lookuptableimportv2_table_handle/
+key_value_init6730_lookuptableimportv2_keys1
-key_value_init6730_lookuptableimportv2_values	
identity??&key_value_init6730/LookupTableImportV2?
&key_value_init6730/LookupTableImportV2LookupTableImportV23key_value_init6730_lookuptableimportv2_table_handle+key_value_init6730_lookuptableimportv2_keys-key_value_init6730_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2(
&key_value_init6730/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityw
NoOpNoOp'^key_value_init6730/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2P
&key_value_init6730/LookupTableImportV2&key_value_init6730/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
:
__inference__creator_13693
identity??
hash_tablez

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name6471*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?#
?
%__inference_model_layer_call_fn_13077
inputs_achievements
inputs_appid
inputs_average_playtime
inputs_categories
inputs_developer
inputs_english
inputs_genres
inputs_median_playtime
inputs_name
inputs_negative_ratings
inputs_owners
inputs_platforms
inputs_positive_ratings
inputs_price
inputs_publisher
inputs_release_date
inputs_required_age
inputs_steamspy_tags
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_achievementsinputs_appidinputs_average_playtimeinputs_categoriesinputs_developerinputs_englishinputs_genresinputs_median_playtimeinputs_nameinputs_negative_ratingsinputs_ownersinputs_platformsinputs_positive_ratingsinputs_priceinputs_publisherinputs_release_dateinputs_required_ageinputs_steamspy_tagsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*1
Tin*
(2&									*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_101662
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/achievements:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/appid:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/average_playtime:ZV
'
_output_shapes
:?????????
+
_user_specified_nameinputs/categories:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/developer:WS
'
_output_shapes
:?????????
(
_user_specified_nameinputs/english:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/genres:_[
'
_output_shapes
:?????????
0
_user_specified_nameinputs/median_playtime:TP
'
_output_shapes
:?????????
%
_user_specified_nameinputs/name:`	\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/negative_ratings:V
R
'
_output_shapes
:?????????
'
_user_specified_nameinputs/owners:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/platforms:`\
'
_output_shapes
:?????????
1
_user_specified_nameinputs/positive_ratings:UQ
'
_output_shapes
:?????????
&
_user_specified_nameinputs/price:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/publisher:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/release_date:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/required_age:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/steamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	
?2
?
__inference__traced_save_14075
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const_29

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const_29"/device:CPU:0*
_output_shapes
 *%
dtypes
2		2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :	:	: :	? @:@:@:: : :	? @:@:@::	? @:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:	: 

_output_shapes
:	:

_output_shapes
: :%	!

_output_shapes
:	? @: 


_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	? @: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	? @: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
??
?
__inference__wrapped_model_9724
achievements	
appid
average_playtime

categories
	developer
english

genres
median_playtime
name
negative_ratings

owners
	platforms
positive_ratings	
price
	publisher
release_date
required_age
steamspy_tagsL
Hmodel_1_model_string_lookup_8_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_8_none_lookup_lookuptablefindv2_default_value	L
Hmodel_1_model_string_lookup_7_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_7_none_lookup_lookuptablefindv2_default_value	L
Hmodel_1_model_string_lookup_6_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_6_none_lookup_lookuptablefindv2_default_value	L
Hmodel_1_model_string_lookup_5_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_5_none_lookup_lookuptablefindv2_default_value	L
Hmodel_1_model_string_lookup_4_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_4_none_lookup_lookuptablefindv2_default_value	L
Hmodel_1_model_string_lookup_3_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_3_none_lookup_lookuptablefindv2_default_value	L
Hmodel_1_model_string_lookup_2_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_2_none_lookup_lookuptablefindv2_default_value	L
Hmodel_1_model_string_lookup_1_none_lookup_lookuptablefindv2_table_handleM
Imodel_1_model_string_lookup_1_none_lookup_lookuptablefindv2_default_value	J
Fmodel_1_model_string_lookup_none_lookup_lookuptablefindv2_table_handleK
Gmodel_1_model_string_lookup_none_lookup_lookuptablefindv2_default_value	%
!model_1_model_normalization_sub_y&
"model_1_model_normalization_sqrt_xJ
7model_1_sequential_dense_matmul_readvariableop_resource:	? @F
8model_1_sequential_dense_biasadd_readvariableop_resource:@K
9model_1_sequential_dense_1_matmul_readvariableop_resource:@H
:model_1_sequential_dense_1_biasadd_readvariableop_resource:
identity??-model_1/model/category_encoding/Assert/Assert?/model_1/model/category_encoding_1/Assert/Assert?/model_1/model/category_encoding_2/Assert/Assert?/model_1/model/category_encoding_3/Assert/Assert?/model_1/model/category_encoding_4/Assert/Assert?/model_1/model/category_encoding_5/Assert/Assert?/model_1/model/category_encoding_6/Assert/Assert?/model_1/model/category_encoding_7/Assert/Assert?/model_1/model/category_encoding_8/Assert/Assert?9model_1/model/string_lookup/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_1/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_2/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_3/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_4/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_5/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_6/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_7/None_Lookup/LookupTableFindV2?;model_1/model/string_lookup_8/None_Lookup/LookupTableFindV2?/model_1/sequential/dense/BiasAdd/ReadVariableOp?.model_1/sequential/dense/MatMul/ReadVariableOp?1model_1/sequential/dense_1/BiasAdd/ReadVariableOp?0model_1/sequential/dense_1/MatMul/ReadVariableOp?
;model_1/model/string_lookup_8/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_8_none_lookup_lookuptablefindv2_table_handleownersImodel_1_model_string_lookup_8_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_8/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_8/IdentityIdentityDmodel_1/model/string_lookup_8/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_8/Identity?
;model_1/model/string_lookup_7/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_7_none_lookup_lookuptablefindv2_table_handlesteamspy_tagsImodel_1_model_string_lookup_7_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_7/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_7/IdentityIdentityDmodel_1/model/string_lookup_7/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_7/Identity?
;model_1/model/string_lookup_6/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_6_none_lookup_lookuptablefindv2_table_handlegenresImodel_1_model_string_lookup_6_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_6/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_6/IdentityIdentityDmodel_1/model/string_lookup_6/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_6/Identity?
;model_1/model/string_lookup_5/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_5_none_lookup_lookuptablefindv2_table_handle
categoriesImodel_1_model_string_lookup_5_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_5/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_5/IdentityIdentityDmodel_1/model/string_lookup_5/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_5/Identity?
;model_1/model/string_lookup_4/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_4_none_lookup_lookuptablefindv2_table_handle	platformsImodel_1_model_string_lookup_4_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_4/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_4/IdentityIdentityDmodel_1/model/string_lookup_4/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_4/Identity?
;model_1/model/string_lookup_3/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_3_none_lookup_lookuptablefindv2_table_handle	publisherImodel_1_model_string_lookup_3_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_3/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_3/IdentityIdentityDmodel_1/model/string_lookup_3/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_3/Identity?
;model_1/model/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_2_none_lookup_lookuptablefindv2_table_handle	developerImodel_1_model_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_2/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_2/IdentityIdentityDmodel_1/model/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_2/Identity?
;model_1/model/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Hmodel_1_model_string_lookup_1_none_lookup_lookuptablefindv2_table_handlerelease_dateImodel_1_model_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2=
;model_1/model/string_lookup_1/None_Lookup/LookupTableFindV2?
&model_1/model/string_lookup_1/IdentityIdentityDmodel_1/model/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2(
&model_1/model/string_lookup_1/Identity?
9model_1/model/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Fmodel_1_model_string_lookup_none_lookup_lookuptablefindv2_table_handlenameGmodel_1_model_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:?????????2;
9model_1/model/string_lookup/None_Lookup/LookupTableFindV2?
$model_1/model/string_lookup/IdentityIdentityBmodel_1/model/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????2&
$model_1/model/string_lookup/Identity?
%model_1/model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_1/model/concatenate/concat/axis?
 model_1/model/concatenate/concatConcatV2appidenglishrequired_ageachievementspositive_ratingsnegative_ratingsaverage_playtimemedian_playtimeprice.model_1/model/concatenate/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????	2"
 model_1/model/concatenate/concat?
model_1/model/normalization/subSub)model_1/model/concatenate/concat:output:0!model_1_model_normalization_sub_y*
T0*'
_output_shapes
:?????????	2!
model_1/model/normalization/sub?
 model_1/model/normalization/SqrtSqrt"model_1_model_normalization_sqrt_x*
T0*
_output_shapes

:	2"
 model_1/model/normalization/Sqrt?
%model_1/model/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%model_1/model/normalization/Maximum/y?
#model_1/model/normalization/MaximumMaximum$model_1/model/normalization/Sqrt:y:0.model_1/model/normalization/Maximum/y:output:0*
T0*
_output_shapes

:	2%
#model_1/model/normalization/Maximum?
#model_1/model/normalization/truedivRealDiv#model_1/model/normalization/sub:z:0'model_1/model/normalization/Maximum:z:0*
T0*'
_output_shapes
:?????????	2%
#model_1/model/normalization/truediv?
%model_1/model/category_encoding/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/model/category_encoding/Const?
#model_1/model/category_encoding/MaxMax-model_1/model/string_lookup/Identity:output:0.model_1/model/category_encoding/Const:output:0*
T0	*
_output_shapes
: 2%
#model_1/model/category_encoding/Max?
'model_1/model/category_encoding/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding/Const_1?
#model_1/model/category_encoding/MinMin-model_1/model/string_lookup/Identity:output:00model_1/model/category_encoding/Const_1:output:0*
T0	*
_output_shapes
: 2%
#model_1/model/category_encoding/Min?
&model_1/model/category_encoding/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2(
&model_1/model/category_encoding/Cast/x?
$model_1/model/category_encoding/CastCast/model_1/model/category_encoding/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2&
$model_1/model/category_encoding/Cast?
'model_1/model/category_encoding/GreaterGreater(model_1/model/category_encoding/Cast:y:0,model_1/model/category_encoding/Max:output:0*
T0	*
_output_shapes
: 2)
'model_1/model/category_encoding/Greater?
(model_1/model/category_encoding/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_1/model/category_encoding/Cast_1/x?
&model_1/model/category_encoding/Cast_1Cast1model_1/model/category_encoding/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding/Cast_1?
,model_1/model/category_encoding/GreaterEqualGreaterEqual,model_1/model/category_encoding/Min:output:0*model_1/model/category_encoding/Cast_1:y:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/GreaterEqual?
*model_1/model/category_encoding/LogicalAnd
LogicalAnd+model_1/model/category_encoding/Greater:z:00model_1/model/category_encoding/GreaterEqual:z:0*
_output_shapes
: 2,
*model_1/model/category_encoding/LogicalAnd?
,model_1/model/category_encoding/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=8502.
,model_1/model/category_encoding/Assert/Const?
4model_1/model/category_encoding/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=85026
4model_1/model/category_encoding/Assert/Assert/data_0?
-model_1/model/category_encoding/Assert/AssertAssert.model_1/model/category_encoding/LogicalAnd:z:0=model_1/model/category_encoding/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2/
-model_1/model/category_encoding/Assert/Assert?
.model_1/model/category_encoding/bincount/ShapeShape-model_1/model/string_lookup/Identity:output:0.^model_1/model/category_encoding/Assert/Assert*
T0	*
_output_shapes
:20
.model_1/model/category_encoding/bincount/Shape?
.model_1/model/category_encoding/bincount/ConstConst.^model_1/model/category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/model/category_encoding/bincount/Const?
-model_1/model/category_encoding/bincount/ProdProd7model_1/model/category_encoding/bincount/Shape:output:07model_1/model/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2/
-model_1/model/category_encoding/bincount/Prod?
2model_1/model/category_encoding/bincount/Greater/yConst.^model_1/model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 24
2model_1/model/category_encoding/bincount/Greater/y?
0model_1/model/category_encoding/bincount/GreaterGreater6model_1/model/category_encoding/bincount/Prod:output:0;model_1/model/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 22
0model_1/model/category_encoding/bincount/Greater?
-model_1/model/category_encoding/bincount/CastCast4model_1/model/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2/
-model_1/model/category_encoding/bincount/Cast?
0model_1/model/category_encoding/bincount/Const_1Const.^model_1/model/category_encoding/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       22
0model_1/model/category_encoding/bincount/Const_1?
,model_1/model/category_encoding/bincount/MaxMax-model_1/model/string_lookup/Identity:output:09model_1/model/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/bincount/Max?
.model_1/model/category_encoding/bincount/add/yConst.^model_1/model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R20
.model_1/model/category_encoding/bincount/add/y?
,model_1/model/category_encoding/bincount/addAddV25model_1/model/category_encoding/bincount/Max:output:07model_1/model/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/bincount/add?
,model_1/model/category_encoding/bincount/mulMul1model_1/model/category_encoding/bincount/Cast:y:00model_1/model/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2.
,model_1/model/category_encoding/bincount/mul?
2model_1/model/category_encoding/bincount/minlengthConst.^model_1/model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?24
2model_1/model/category_encoding/bincount/minlength?
0model_1/model/category_encoding/bincount/MaximumMaximum;model_1/model/category_encoding/bincount/minlength:output:00model_1/model/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 22
0model_1/model/category_encoding/bincount/Maximum?
2model_1/model/category_encoding/bincount/maxlengthConst.^model_1/model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?24
2model_1/model/category_encoding/bincount/maxlength?
0model_1/model/category_encoding/bincount/MinimumMinimum;model_1/model/category_encoding/bincount/maxlength:output:04model_1/model/category_encoding/bincount/Maximum:z:0*
T0	*
_output_shapes
: 22
0model_1/model/category_encoding/bincount/Minimum?
0model_1/model/category_encoding/bincount/Const_2Const.^model_1/model/category_encoding/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 22
0model_1/model/category_encoding/bincount/Const_2?
6model_1/model/category_encoding/bincount/DenseBincountDenseBincount-model_1/model/string_lookup/Identity:output:04model_1/model/category_encoding/bincount/Minimum:z:09model_1/model/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(28
6model_1/model/category_encoding/bincount/DenseBincount?
'model_1/model/category_encoding_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_1/Const?
%model_1/model/category_encoding_1/MaxMax/model_1/model/string_lookup_1/Identity:output:00model_1/model/category_encoding_1/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_1/Max?
)model_1/model/category_encoding_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_1/Const_1?
%model_1/model/category_encoding_1/MinMin/model_1/model/string_lookup_1/Identity:output:02model_1/model/category_encoding_1/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_1/Min?
(model_1/model/category_encoding_1/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2*
(model_1/model/category_encoding_1/Cast/x?
&model_1/model/category_encoding_1/CastCast1model_1/model/category_encoding_1/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_1/Cast?
)model_1/model/category_encoding_1/GreaterGreater*model_1/model/category_encoding_1/Cast:y:0.model_1/model/category_encoding_1/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_1/Greater?
*model_1/model/category_encoding_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_1/Cast_1/x?
(model_1/model/category_encoding_1/Cast_1Cast3model_1/model/category_encoding_1/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_1/Cast_1?
.model_1/model/category_encoding_1/GreaterEqualGreaterEqual.model_1/model/category_encoding_1/Min:output:0,model_1/model/category_encoding_1/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_1/GreaterEqual?
,model_1/model/category_encoding_1/LogicalAnd
LogicalAnd-model_1/model/category_encoding_1/Greater:z:02model_1/model/category_encoding_1/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_1/LogicalAnd?
.model_1/model/category_encoding_1/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=68520
.model_1/model/category_encoding_1/Assert/Const?
6model_1/model/category_encoding_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=68528
6model_1/model/category_encoding_1/Assert/Assert/data_0?
/model_1/model/category_encoding_1/Assert/AssertAssert0model_1/model/category_encoding_1/LogicalAnd:z:0?model_1/model/category_encoding_1/Assert/Assert/data_0:output:0.^model_1/model/category_encoding/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_1/Assert/Assert?
0model_1/model/category_encoding_1/bincount/ShapeShape/model_1/model/string_lookup_1/Identity:output:00^model_1/model/category_encoding_1/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_1/bincount/Shape?
0model_1/model/category_encoding_1/bincount/ConstConst0^model_1/model/category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_1/bincount/Const?
/model_1/model/category_encoding_1/bincount/ProdProd9model_1/model/category_encoding_1/bincount/Shape:output:09model_1/model/category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_1/bincount/Prod?
4model_1/model/category_encoding_1/bincount/Greater/yConst0^model_1/model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_1/bincount/Greater/y?
2model_1/model/category_encoding_1/bincount/GreaterGreater8model_1/model/category_encoding_1/bincount/Prod:output:0=model_1/model/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_1/bincount/Greater?
/model_1/model/category_encoding_1/bincount/CastCast6model_1/model/category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_1/bincount/Cast?
2model_1/model/category_encoding_1/bincount/Const_1Const0^model_1/model/category_encoding_1/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_1/bincount/Const_1?
.model_1/model/category_encoding_1/bincount/MaxMax/model_1/model/string_lookup_1/Identity:output:0;model_1/model/category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_1/bincount/Max?
0model_1/model/category_encoding_1/bincount/add/yConst0^model_1/model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_1/bincount/add/y?
.model_1/model/category_encoding_1/bincount/addAddV27model_1/model/category_encoding_1/bincount/Max:output:09model_1/model/category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_1/bincount/add?
.model_1/model/category_encoding_1/bincount/mulMul3model_1/model/category_encoding_1/bincount/Cast:y:02model_1/model/category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_1/bincount/mul?
4model_1/model/category_encoding_1/bincount/minlengthConst0^model_1/model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_1/bincount/minlength?
2model_1/model/category_encoding_1/bincount/MaximumMaximum=model_1/model/category_encoding_1/bincount/minlength:output:02model_1/model/category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_1/bincount/Maximum?
4model_1/model/category_encoding_1/bincount/maxlengthConst0^model_1/model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_1/bincount/maxlength?
2model_1/model/category_encoding_1/bincount/MinimumMinimum=model_1/model/category_encoding_1/bincount/maxlength:output:06model_1/model/category_encoding_1/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_1/bincount/Minimum?
2model_1/model/category_encoding_1/bincount/Const_2Const0^model_1/model/category_encoding_1/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_1/bincount/Const_2?
8model_1/model/category_encoding_1/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_1/Identity:output:06model_1/model/category_encoding_1/bincount/Minimum:z:0;model_1/model/category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2:
8model_1/model/category_encoding_1/bincount/DenseBincount?
'model_1/model/category_encoding_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_2/Const?
%model_1/model/category_encoding_2/MaxMax/model_1/model/string_lookup_2/Identity:output:00model_1/model/category_encoding_2/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_2/Max?
)model_1/model/category_encoding_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_2/Const_1?
%model_1/model/category_encoding_2/MinMin/model_1/model/string_lookup_2/Identity:output:02model_1/model/category_encoding_2/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_2/Min?
(model_1/model/category_encoding_2/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2*
(model_1/model/category_encoding_2/Cast/x?
&model_1/model/category_encoding_2/CastCast1model_1/model/category_encoding_2/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_2/Cast?
)model_1/model/category_encoding_2/GreaterGreater*model_1/model/category_encoding_2/Cast:y:0.model_1/model/category_encoding_2/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_2/Greater?
*model_1/model/category_encoding_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_2/Cast_1/x?
(model_1/model/category_encoding_2/Cast_1Cast3model_1/model/category_encoding_2/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_2/Cast_1?
.model_1/model/category_encoding_2/GreaterEqualGreaterEqual.model_1/model/category_encoding_2/Min:output:0,model_1/model/category_encoding_2/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_2/GreaterEqual?
,model_1/model/category_encoding_2/LogicalAnd
LogicalAnd-model_1/model/category_encoding_2/Greater:z:02model_1/model/category_encoding_2/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_2/LogicalAnd?
.model_1/model/category_encoding_2/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=65420
.model_1/model/category_encoding_2/Assert/Const?
6model_1/model/category_encoding_2/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=65428
6model_1/model/category_encoding_2/Assert/Assert/data_0?
/model_1/model/category_encoding_2/Assert/AssertAssert0model_1/model/category_encoding_2/LogicalAnd:z:0?model_1/model/category_encoding_2/Assert/Assert/data_0:output:00^model_1/model/category_encoding_1/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_2/Assert/Assert?
0model_1/model/category_encoding_2/bincount/ShapeShape/model_1/model/string_lookup_2/Identity:output:00^model_1/model/category_encoding_2/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_2/bincount/Shape?
0model_1/model/category_encoding_2/bincount/ConstConst0^model_1/model/category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_2/bincount/Const?
/model_1/model/category_encoding_2/bincount/ProdProd9model_1/model/category_encoding_2/bincount/Shape:output:09model_1/model/category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_2/bincount/Prod?
4model_1/model/category_encoding_2/bincount/Greater/yConst0^model_1/model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_2/bincount/Greater/y?
2model_1/model/category_encoding_2/bincount/GreaterGreater8model_1/model/category_encoding_2/bincount/Prod:output:0=model_1/model/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_2/bincount/Greater?
/model_1/model/category_encoding_2/bincount/CastCast6model_1/model/category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_2/bincount/Cast?
2model_1/model/category_encoding_2/bincount/Const_1Const0^model_1/model/category_encoding_2/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_2/bincount/Const_1?
.model_1/model/category_encoding_2/bincount/MaxMax/model_1/model/string_lookup_2/Identity:output:0;model_1/model/category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_2/bincount/Max?
0model_1/model/category_encoding_2/bincount/add/yConst0^model_1/model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_2/bincount/add/y?
.model_1/model/category_encoding_2/bincount/addAddV27model_1/model/category_encoding_2/bincount/Max:output:09model_1/model/category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_2/bincount/add?
.model_1/model/category_encoding_2/bincount/mulMul3model_1/model/category_encoding_2/bincount/Cast:y:02model_1/model/category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_2/bincount/mul?
4model_1/model/category_encoding_2/bincount/minlengthConst0^model_1/model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_2/bincount/minlength?
2model_1/model/category_encoding_2/bincount/MaximumMaximum=model_1/model/category_encoding_2/bincount/minlength:output:02model_1/model/category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_2/bincount/Maximum?
4model_1/model/category_encoding_2/bincount/maxlengthConst0^model_1/model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_2/bincount/maxlength?
2model_1/model/category_encoding_2/bincount/MinimumMinimum=model_1/model/category_encoding_2/bincount/maxlength:output:06model_1/model/category_encoding_2/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_2/bincount/Minimum?
2model_1/model/category_encoding_2/bincount/Const_2Const0^model_1/model/category_encoding_2/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_2/bincount/Const_2?
8model_1/model/category_encoding_2/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_2/Identity:output:06model_1/model/category_encoding_2/bincount/Minimum:z:0;model_1/model/category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2:
8model_1/model/category_encoding_2/bincount/DenseBincount?
'model_1/model/category_encoding_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_3/Const?
%model_1/model/category_encoding_3/MaxMax/model_1/model/string_lookup_3/Identity:output:00model_1/model/category_encoding_3/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_3/Max?
)model_1/model/category_encoding_3/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_3/Const_1?
%model_1/model/category_encoding_3/MinMin/model_1/model/string_lookup_3/Identity:output:02model_1/model/category_encoding_3/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_3/Min?
(model_1/model/category_encoding_3/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2*
(model_1/model/category_encoding_3/Cast/x?
&model_1/model/category_encoding_3/CastCast1model_1/model/category_encoding_3/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_3/Cast?
)model_1/model/category_encoding_3/GreaterGreater*model_1/model/category_encoding_3/Cast:y:0.model_1/model/category_encoding_3/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_3/Greater?
*model_1/model/category_encoding_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_3/Cast_1/x?
(model_1/model/category_encoding_3/Cast_1Cast3model_1/model/category_encoding_3/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_3/Cast_1?
.model_1/model/category_encoding_3/GreaterEqualGreaterEqual.model_1/model/category_encoding_3/Min:output:0,model_1/model/category_encoding_3/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_3/GreaterEqual?
,model_1/model/category_encoding_3/LogicalAnd
LogicalAnd-model_1/model/category_encoding_3/Greater:z:02model_1/model/category_encoding_3/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_3/LogicalAnd?
.model_1/model/category_encoding_3/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=47920
.model_1/model/category_encoding_3/Assert/Const?
6model_1/model/category_encoding_3/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=47928
6model_1/model/category_encoding_3/Assert/Assert/data_0?
/model_1/model/category_encoding_3/Assert/AssertAssert0model_1/model/category_encoding_3/LogicalAnd:z:0?model_1/model/category_encoding_3/Assert/Assert/data_0:output:00^model_1/model/category_encoding_2/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_3/Assert/Assert?
0model_1/model/category_encoding_3/bincount/ShapeShape/model_1/model/string_lookup_3/Identity:output:00^model_1/model/category_encoding_3/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_3/bincount/Shape?
0model_1/model/category_encoding_3/bincount/ConstConst0^model_1/model/category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_3/bincount/Const?
/model_1/model/category_encoding_3/bincount/ProdProd9model_1/model/category_encoding_3/bincount/Shape:output:09model_1/model/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_3/bincount/Prod?
4model_1/model/category_encoding_3/bincount/Greater/yConst0^model_1/model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_3/bincount/Greater/y?
2model_1/model/category_encoding_3/bincount/GreaterGreater8model_1/model/category_encoding_3/bincount/Prod:output:0=model_1/model/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_3/bincount/Greater?
/model_1/model/category_encoding_3/bincount/CastCast6model_1/model/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_3/bincount/Cast?
2model_1/model/category_encoding_3/bincount/Const_1Const0^model_1/model/category_encoding_3/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_3/bincount/Const_1?
.model_1/model/category_encoding_3/bincount/MaxMax/model_1/model/string_lookup_3/Identity:output:0;model_1/model/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_3/bincount/Max?
0model_1/model/category_encoding_3/bincount/add/yConst0^model_1/model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_3/bincount/add/y?
.model_1/model/category_encoding_3/bincount/addAddV27model_1/model/category_encoding_3/bincount/Max:output:09model_1/model/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_3/bincount/add?
.model_1/model/category_encoding_3/bincount/mulMul3model_1/model/category_encoding_3/bincount/Cast:y:02model_1/model/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_3/bincount/mul?
4model_1/model/category_encoding_3/bincount/minlengthConst0^model_1/model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_3/bincount/minlength?
2model_1/model/category_encoding_3/bincount/MaximumMaximum=model_1/model/category_encoding_3/bincount/minlength:output:02model_1/model/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_3/bincount/Maximum?
4model_1/model/category_encoding_3/bincount/maxlengthConst0^model_1/model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_3/bincount/maxlength?
2model_1/model/category_encoding_3/bincount/MinimumMinimum=model_1/model/category_encoding_3/bincount/maxlength:output:06model_1/model/category_encoding_3/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_3/bincount/Minimum?
2model_1/model/category_encoding_3/bincount/Const_2Const0^model_1/model/category_encoding_3/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_3/bincount/Const_2?
8model_1/model/category_encoding_3/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_3/Identity:output:06model_1/model/category_encoding_3/bincount/Minimum:z:0;model_1/model/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2:
8model_1/model/category_encoding_3/bincount/DenseBincount?
'model_1/model/category_encoding_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_4/Const?
%model_1/model/category_encoding_4/MaxMax/model_1/model/string_lookup_4/Identity:output:00model_1/model/category_encoding_4/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_4/Max?
)model_1/model/category_encoding_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_4/Const_1?
%model_1/model/category_encoding_4/MinMin/model_1/model/string_lookup_4/Identity:output:02model_1/model/category_encoding_4/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_4/Min?
(model_1/model/category_encoding_4/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_1/model/category_encoding_4/Cast/x?
&model_1/model/category_encoding_4/CastCast1model_1/model/category_encoding_4/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_4/Cast?
)model_1/model/category_encoding_4/GreaterGreater*model_1/model/category_encoding_4/Cast:y:0.model_1/model/category_encoding_4/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_4/Greater?
*model_1/model/category_encoding_4/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_4/Cast_1/x?
(model_1/model/category_encoding_4/Cast_1Cast3model_1/model/category_encoding_4/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_4/Cast_1?
.model_1/model/category_encoding_4/GreaterEqualGreaterEqual.model_1/model/category_encoding_4/Min:output:0,model_1/model/category_encoding_4/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_4/GreaterEqual?
,model_1/model/category_encoding_4/LogicalAnd
LogicalAnd-model_1/model/category_encoding_4/Greater:z:02model_1/model/category_encoding_4/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_4/LogicalAnd?
.model_1/model/category_encoding_4/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=520
.model_1/model/category_encoding_4/Assert/Const?
6model_1/model/category_encoding_4/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=528
6model_1/model/category_encoding_4/Assert/Assert/data_0?
/model_1/model/category_encoding_4/Assert/AssertAssert0model_1/model/category_encoding_4/LogicalAnd:z:0?model_1/model/category_encoding_4/Assert/Assert/data_0:output:00^model_1/model/category_encoding_3/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_4/Assert/Assert?
0model_1/model/category_encoding_4/bincount/ShapeShape/model_1/model/string_lookup_4/Identity:output:00^model_1/model/category_encoding_4/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_4/bincount/Shape?
0model_1/model/category_encoding_4/bincount/ConstConst0^model_1/model/category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_4/bincount/Const?
/model_1/model/category_encoding_4/bincount/ProdProd9model_1/model/category_encoding_4/bincount/Shape:output:09model_1/model/category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_4/bincount/Prod?
4model_1/model/category_encoding_4/bincount/Greater/yConst0^model_1/model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_4/bincount/Greater/y?
2model_1/model/category_encoding_4/bincount/GreaterGreater8model_1/model/category_encoding_4/bincount/Prod:output:0=model_1/model/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_4/bincount/Greater?
/model_1/model/category_encoding_4/bincount/CastCast6model_1/model/category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_4/bincount/Cast?
2model_1/model/category_encoding_4/bincount/Const_1Const0^model_1/model/category_encoding_4/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_4/bincount/Const_1?
.model_1/model/category_encoding_4/bincount/MaxMax/model_1/model/string_lookup_4/Identity:output:0;model_1/model/category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_4/bincount/Max?
0model_1/model/category_encoding_4/bincount/add/yConst0^model_1/model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_4/bincount/add/y?
.model_1/model/category_encoding_4/bincount/addAddV27model_1/model/category_encoding_4/bincount/Max:output:09model_1/model/category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_4/bincount/add?
.model_1/model/category_encoding_4/bincount/mulMul3model_1/model/category_encoding_4/bincount/Cast:y:02model_1/model/category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_4/bincount/mul?
4model_1/model/category_encoding_4/bincount/minlengthConst0^model_1/model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R26
4model_1/model/category_encoding_4/bincount/minlength?
2model_1/model/category_encoding_4/bincount/MaximumMaximum=model_1/model/category_encoding_4/bincount/minlength:output:02model_1/model/category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_4/bincount/Maximum?
4model_1/model/category_encoding_4/bincount/maxlengthConst0^model_1/model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R26
4model_1/model/category_encoding_4/bincount/maxlength?
2model_1/model/category_encoding_4/bincount/MinimumMinimum=model_1/model/category_encoding_4/bincount/maxlength:output:06model_1/model/category_encoding_4/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_4/bincount/Minimum?
2model_1/model/category_encoding_4/bincount/Const_2Const0^model_1/model/category_encoding_4/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_4/bincount/Const_2?
8model_1/model/category_encoding_4/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_4/Identity:output:06model_1/model/category_encoding_4/bincount/Minimum:z:0;model_1/model/category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2:
8model_1/model/category_encoding_4/bincount/DenseBincount?
'model_1/model/category_encoding_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_5/Const?
%model_1/model/category_encoding_5/MaxMax/model_1/model/string_lookup_5/Identity:output:00model_1/model/category_encoding_5/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_5/Max?
)model_1/model/category_encoding_5/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_5/Const_1?
%model_1/model/category_encoding_5/MinMin/model_1/model/string_lookup_5/Identity:output:02model_1/model/category_encoding_5/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_5/Min?
(model_1/model/category_encoding_5/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2*
(model_1/model/category_encoding_5/Cast/x?
&model_1/model/category_encoding_5/CastCast1model_1/model/category_encoding_5/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_5/Cast?
)model_1/model/category_encoding_5/GreaterGreater*model_1/model/category_encoding_5/Cast:y:0.model_1/model/category_encoding_5/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_5/Greater?
*model_1/model/category_encoding_5/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_5/Cast_1/x?
(model_1/model/category_encoding_5/Cast_1Cast3model_1/model/category_encoding_5/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_5/Cast_1?
.model_1/model/category_encoding_5/GreaterEqualGreaterEqual.model_1/model/category_encoding_5/Min:output:0,model_1/model/category_encoding_5/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_5/GreaterEqual?
,model_1/model/category_encoding_5/LogicalAnd
LogicalAnd-model_1/model/category_encoding_5/Greater:z:02model_1/model/category_encoding_5/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_5/LogicalAnd?
.model_1/model/category_encoding_5/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=49520
.model_1/model/category_encoding_5/Assert/Const?
6model_1/model/category_encoding_5/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=49528
6model_1/model/category_encoding_5/Assert/Assert/data_0?
/model_1/model/category_encoding_5/Assert/AssertAssert0model_1/model/category_encoding_5/LogicalAnd:z:0?model_1/model/category_encoding_5/Assert/Assert/data_0:output:00^model_1/model/category_encoding_4/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_5/Assert/Assert?
0model_1/model/category_encoding_5/bincount/ShapeShape/model_1/model/string_lookup_5/Identity:output:00^model_1/model/category_encoding_5/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_5/bincount/Shape?
0model_1/model/category_encoding_5/bincount/ConstConst0^model_1/model/category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_5/bincount/Const?
/model_1/model/category_encoding_5/bincount/ProdProd9model_1/model/category_encoding_5/bincount/Shape:output:09model_1/model/category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_5/bincount/Prod?
4model_1/model/category_encoding_5/bincount/Greater/yConst0^model_1/model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_5/bincount/Greater/y?
2model_1/model/category_encoding_5/bincount/GreaterGreater8model_1/model/category_encoding_5/bincount/Prod:output:0=model_1/model/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_5/bincount/Greater?
/model_1/model/category_encoding_5/bincount/CastCast6model_1/model/category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_5/bincount/Cast?
2model_1/model/category_encoding_5/bincount/Const_1Const0^model_1/model/category_encoding_5/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_5/bincount/Const_1?
.model_1/model/category_encoding_5/bincount/MaxMax/model_1/model/string_lookup_5/Identity:output:0;model_1/model/category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_5/bincount/Max?
0model_1/model/category_encoding_5/bincount/add/yConst0^model_1/model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_5/bincount/add/y?
.model_1/model/category_encoding_5/bincount/addAddV27model_1/model/category_encoding_5/bincount/Max:output:09model_1/model/category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_5/bincount/add?
.model_1/model/category_encoding_5/bincount/mulMul3model_1/model/category_encoding_5/bincount/Cast:y:02model_1/model/category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_5/bincount/mul?
4model_1/model/category_encoding_5/bincount/minlengthConst0^model_1/model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_5/bincount/minlength?
2model_1/model/category_encoding_5/bincount/MaximumMaximum=model_1/model/category_encoding_5/bincount/minlength:output:02model_1/model/category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_5/bincount/Maximum?
4model_1/model/category_encoding_5/bincount/maxlengthConst0^model_1/model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_5/bincount/maxlength?
2model_1/model/category_encoding_5/bincount/MinimumMinimum=model_1/model/category_encoding_5/bincount/maxlength:output:06model_1/model/category_encoding_5/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_5/bincount/Minimum?
2model_1/model/category_encoding_5/bincount/Const_2Const0^model_1/model/category_encoding_5/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_5/bincount/Const_2?
8model_1/model/category_encoding_5/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_5/Identity:output:06model_1/model/category_encoding_5/bincount/Minimum:z:0;model_1/model/category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2:
8model_1/model/category_encoding_5/bincount/DenseBincount?
'model_1/model/category_encoding_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_6/Const?
%model_1/model/category_encoding_6/MaxMax/model_1/model/string_lookup_6/Identity:output:00model_1/model/category_encoding_6/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_6/Max?
)model_1/model/category_encoding_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_6/Const_1?
%model_1/model/category_encoding_6/MinMin/model_1/model/string_lookup_6/Identity:output:02model_1/model/category_encoding_6/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_6/Min?
(model_1/model/category_encoding_6/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2*
(model_1/model/category_encoding_6/Cast/x?
&model_1/model/category_encoding_6/CastCast1model_1/model/category_encoding_6/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_6/Cast?
)model_1/model/category_encoding_6/GreaterGreater*model_1/model/category_encoding_6/Cast:y:0.model_1/model/category_encoding_6/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_6/Greater?
*model_1/model/category_encoding_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_6/Cast_1/x?
(model_1/model/category_encoding_6/Cast_1Cast3model_1/model/category_encoding_6/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_6/Cast_1?
.model_1/model/category_encoding_6/GreaterEqualGreaterEqual.model_1/model/category_encoding_6/Min:output:0,model_1/model/category_encoding_6/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_6/GreaterEqual?
,model_1/model/category_encoding_6/LogicalAnd
LogicalAnd-model_1/model/category_encoding_6/Greater:z:02model_1/model/category_encoding_6/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_6/LogicalAnd?
.model_1/model/category_encoding_6/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=21720
.model_1/model/category_encoding_6/Assert/Const?
6model_1/model/category_encoding_6/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=21728
6model_1/model/category_encoding_6/Assert/Assert/data_0?
/model_1/model/category_encoding_6/Assert/AssertAssert0model_1/model/category_encoding_6/LogicalAnd:z:0?model_1/model/category_encoding_6/Assert/Assert/data_0:output:00^model_1/model/category_encoding_5/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_6/Assert/Assert?
0model_1/model/category_encoding_6/bincount/ShapeShape/model_1/model/string_lookup_6/Identity:output:00^model_1/model/category_encoding_6/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_6/bincount/Shape?
0model_1/model/category_encoding_6/bincount/ConstConst0^model_1/model/category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_6/bincount/Const?
/model_1/model/category_encoding_6/bincount/ProdProd9model_1/model/category_encoding_6/bincount/Shape:output:09model_1/model/category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_6/bincount/Prod?
4model_1/model/category_encoding_6/bincount/Greater/yConst0^model_1/model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_6/bincount/Greater/y?
2model_1/model/category_encoding_6/bincount/GreaterGreater8model_1/model/category_encoding_6/bincount/Prod:output:0=model_1/model/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_6/bincount/Greater?
/model_1/model/category_encoding_6/bincount/CastCast6model_1/model/category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_6/bincount/Cast?
2model_1/model/category_encoding_6/bincount/Const_1Const0^model_1/model/category_encoding_6/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_6/bincount/Const_1?
.model_1/model/category_encoding_6/bincount/MaxMax/model_1/model/string_lookup_6/Identity:output:0;model_1/model/category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_6/bincount/Max?
0model_1/model/category_encoding_6/bincount/add/yConst0^model_1/model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_6/bincount/add/y?
.model_1/model/category_encoding_6/bincount/addAddV27model_1/model/category_encoding_6/bincount/Max:output:09model_1/model/category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_6/bincount/add?
.model_1/model/category_encoding_6/bincount/mulMul3model_1/model/category_encoding_6/bincount/Cast:y:02model_1/model/category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_6/bincount/mul?
4model_1/model/category_encoding_6/bincount/minlengthConst0^model_1/model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_6/bincount/minlength?
2model_1/model/category_encoding_6/bincount/MaximumMaximum=model_1/model/category_encoding_6/bincount/minlength:output:02model_1/model/category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_6/bincount/Maximum?
4model_1/model/category_encoding_6/bincount/maxlengthConst0^model_1/model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_6/bincount/maxlength?
2model_1/model/category_encoding_6/bincount/MinimumMinimum=model_1/model/category_encoding_6/bincount/maxlength:output:06model_1/model/category_encoding_6/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_6/bincount/Minimum?
2model_1/model/category_encoding_6/bincount/Const_2Const0^model_1/model/category_encoding_6/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_6/bincount/Const_2?
8model_1/model/category_encoding_6/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_6/Identity:output:06model_1/model/category_encoding_6/bincount/Minimum:z:0;model_1/model/category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2:
8model_1/model/category_encoding_6/bincount/DenseBincount?
'model_1/model/category_encoding_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_7/Const?
%model_1/model/category_encoding_7/MaxMax/model_1/model/string_lookup_7/Identity:output:00model_1/model/category_encoding_7/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_7/Max?
)model_1/model/category_encoding_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_7/Const_1?
%model_1/model/category_encoding_7/MinMin/model_1/model/string_lookup_7/Identity:output:02model_1/model/category_encoding_7/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_7/Min?
(model_1/model/category_encoding_7/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :?2*
(model_1/model/category_encoding_7/Cast/x?
&model_1/model/category_encoding_7/CastCast1model_1/model/category_encoding_7/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_7/Cast?
)model_1/model/category_encoding_7/GreaterGreater*model_1/model/category_encoding_7/Cast:y:0.model_1/model/category_encoding_7/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_7/Greater?
*model_1/model/category_encoding_7/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_7/Cast_1/x?
(model_1/model/category_encoding_7/Cast_1Cast3model_1/model/category_encoding_7/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_7/Cast_1?
.model_1/model/category_encoding_7/GreaterEqualGreaterEqual.model_1/model/category_encoding_7/Min:output:0,model_1/model/category_encoding_7/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_7/GreaterEqual?
,model_1/model/category_encoding_7/LogicalAnd
LogicalAnd-model_1/model/category_encoding_7/Greater:z:02model_1/model/category_encoding_7/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_7/LogicalAnd?
.model_1/model/category_encoding_7/Assert/ConstConst*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=74420
.model_1/model/category_encoding_7/Assert/Const?
6model_1/model/category_encoding_7/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*_
valueVBT BNInput values must be in the range 0 <= values < num_tokens with num_tokens=74428
6model_1/model/category_encoding_7/Assert/Assert/data_0?
/model_1/model/category_encoding_7/Assert/AssertAssert0model_1/model/category_encoding_7/LogicalAnd:z:0?model_1/model/category_encoding_7/Assert/Assert/data_0:output:00^model_1/model/category_encoding_6/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_7/Assert/Assert?
0model_1/model/category_encoding_7/bincount/ShapeShape/model_1/model/string_lookup_7/Identity:output:00^model_1/model/category_encoding_7/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_7/bincount/Shape?
0model_1/model/category_encoding_7/bincount/ConstConst0^model_1/model/category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_7/bincount/Const?
/model_1/model/category_encoding_7/bincount/ProdProd9model_1/model/category_encoding_7/bincount/Shape:output:09model_1/model/category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_7/bincount/Prod?
4model_1/model/category_encoding_7/bincount/Greater/yConst0^model_1/model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_7/bincount/Greater/y?
2model_1/model/category_encoding_7/bincount/GreaterGreater8model_1/model/category_encoding_7/bincount/Prod:output:0=model_1/model/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_7/bincount/Greater?
/model_1/model/category_encoding_7/bincount/CastCast6model_1/model/category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_7/bincount/Cast?
2model_1/model/category_encoding_7/bincount/Const_1Const0^model_1/model/category_encoding_7/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_7/bincount/Const_1?
.model_1/model/category_encoding_7/bincount/MaxMax/model_1/model/string_lookup_7/Identity:output:0;model_1/model/category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_7/bincount/Max?
0model_1/model/category_encoding_7/bincount/add/yConst0^model_1/model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_7/bincount/add/y?
.model_1/model/category_encoding_7/bincount/addAddV27model_1/model/category_encoding_7/bincount/Max:output:09model_1/model/category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_7/bincount/add?
.model_1/model/category_encoding_7/bincount/mulMul3model_1/model/category_encoding_7/bincount/Cast:y:02model_1/model/category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_7/bincount/mul?
4model_1/model/category_encoding_7/bincount/minlengthConst0^model_1/model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_7/bincount/minlength?
2model_1/model/category_encoding_7/bincount/MaximumMaximum=model_1/model/category_encoding_7/bincount/minlength:output:02model_1/model/category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_7/bincount/Maximum?
4model_1/model/category_encoding_7/bincount/maxlengthConst0^model_1/model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0	*
value
B	 R?26
4model_1/model/category_encoding_7/bincount/maxlength?
2model_1/model/category_encoding_7/bincount/MinimumMinimum=model_1/model/category_encoding_7/bincount/maxlength:output:06model_1/model/category_encoding_7/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_7/bincount/Minimum?
2model_1/model/category_encoding_7/bincount/Const_2Const0^model_1/model/category_encoding_7/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_7/bincount/Const_2?
8model_1/model/category_encoding_7/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_7/Identity:output:06model_1/model/category_encoding_7/bincount/Minimum:z:0;model_1/model/category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*(
_output_shapes
:??????????*
binary_output(2:
8model_1/model/category_encoding_7/bincount/DenseBincount?
'model_1/model/category_encoding_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/model/category_encoding_8/Const?
%model_1/model/category_encoding_8/MaxMax/model_1/model/string_lookup_8/Identity:output:00model_1/model/category_encoding_8/Const:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_8/Max?
)model_1/model/category_encoding_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/model/category_encoding_8/Const_1?
%model_1/model/category_encoding_8/MinMin/model_1/model/string_lookup_8/Identity:output:02model_1/model/category_encoding_8/Const_1:output:0*
T0	*
_output_shapes
: 2'
%model_1/model/category_encoding_8/Min?
(model_1/model/category_encoding_8/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2*
(model_1/model/category_encoding_8/Cast/x?
&model_1/model/category_encoding_8/CastCast1model_1/model/category_encoding_8/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2(
&model_1/model/category_encoding_8/Cast?
)model_1/model/category_encoding_8/GreaterGreater*model_1/model/category_encoding_8/Cast:y:0.model_1/model/category_encoding_8/Max:output:0*
T0	*
_output_shapes
: 2+
)model_1/model/category_encoding_8/Greater?
*model_1/model/category_encoding_8/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : 2,
*model_1/model/category_encoding_8/Cast_1/x?
(model_1/model/category_encoding_8/Cast_1Cast3model_1/model/category_encoding_8/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2*
(model_1/model/category_encoding_8/Cast_1?
.model_1/model/category_encoding_8/GreaterEqualGreaterEqual.model_1/model/category_encoding_8/Min:output:0,model_1/model/category_encoding_8/Cast_1:y:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_8/GreaterEqual?
,model_1/model/category_encoding_8/LogicalAnd
LogicalAnd-model_1/model/category_encoding_8/Greater:z:02model_1/model/category_encoding_8/GreaterEqual:z:0*
_output_shapes
: 2.
,model_1/model/category_encoding_8/LogicalAnd?
.model_1/model/category_encoding_8/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=1120
.model_1/model/category_encoding_8/Assert/Const?
6model_1/model/category_encoding_8/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=1128
6model_1/model/category_encoding_8/Assert/Assert/data_0?
/model_1/model/category_encoding_8/Assert/AssertAssert0model_1/model/category_encoding_8/LogicalAnd:z:0?model_1/model/category_encoding_8/Assert/Assert/data_0:output:00^model_1/model/category_encoding_7/Assert/Assert*

T
2*
_output_shapes
 21
/model_1/model/category_encoding_8/Assert/Assert?
0model_1/model/category_encoding_8/bincount/ShapeShape/model_1/model/string_lookup_8/Identity:output:00^model_1/model/category_encoding_8/Assert/Assert*
T0	*
_output_shapes
:22
0model_1/model/category_encoding_8/bincount/Shape?
0model_1/model/category_encoding_8/bincount/ConstConst0^model_1/model/category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/model/category_encoding_8/bincount/Const?
/model_1/model/category_encoding_8/bincount/ProdProd9model_1/model/category_encoding_8/bincount/Shape:output:09model_1/model/category_encoding_8/bincount/Const:output:0*
T0*
_output_shapes
: 21
/model_1/model/category_encoding_8/bincount/Prod?
4model_1/model/category_encoding_8/bincount/Greater/yConst0^model_1/model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : 26
4model_1/model/category_encoding_8/bincount/Greater/y?
2model_1/model/category_encoding_8/bincount/GreaterGreater8model_1/model/category_encoding_8/bincount/Prod:output:0=model_1/model/category_encoding_8/bincount/Greater/y:output:0*
T0*
_output_shapes
: 24
2model_1/model/category_encoding_8/bincount/Greater?
/model_1/model/category_encoding_8/bincount/CastCast6model_1/model/category_encoding_8/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 21
/model_1/model/category_encoding_8/bincount/Cast?
2model_1/model/category_encoding_8/bincount/Const_1Const0^model_1/model/category_encoding_8/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       24
2model_1/model/category_encoding_8/bincount/Const_1?
.model_1/model/category_encoding_8/bincount/MaxMax/model_1/model/string_lookup_8/Identity:output:0;model_1/model/category_encoding_8/bincount/Const_1:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_8/bincount/Max?
0model_1/model/category_encoding_8/bincount/add/yConst0^model_1/model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R22
0model_1/model/category_encoding_8/bincount/add/y?
.model_1/model/category_encoding_8/bincount/addAddV27model_1/model/category_encoding_8/bincount/Max:output:09model_1/model/category_encoding_8/bincount/add/y:output:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_8/bincount/add?
.model_1/model/category_encoding_8/bincount/mulMul3model_1/model/category_encoding_8/bincount/Cast:y:02model_1/model/category_encoding_8/bincount/add:z:0*
T0	*
_output_shapes
: 20
.model_1/model/category_encoding_8/bincount/mul?
4model_1/model/category_encoding_8/bincount/minlengthConst0^model_1/model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R26
4model_1/model/category_encoding_8/bincount/minlength?
2model_1/model/category_encoding_8/bincount/MaximumMaximum=model_1/model/category_encoding_8/bincount/minlength:output:02model_1/model/category_encoding_8/bincount/mul:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_8/bincount/Maximum?
4model_1/model/category_encoding_8/bincount/maxlengthConst0^model_1/model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R26
4model_1/model/category_encoding_8/bincount/maxlength?
2model_1/model/category_encoding_8/bincount/MinimumMinimum=model_1/model/category_encoding_8/bincount/maxlength:output:06model_1/model/category_encoding_8/bincount/Maximum:z:0*
T0	*
_output_shapes
: 24
2model_1/model/category_encoding_8/bincount/Minimum?
2model_1/model/category_encoding_8/bincount/Const_2Const0^model_1/model/category_encoding_8/Assert/Assert*
_output_shapes
: *
dtype0*
valueB 24
2model_1/model/category_encoding_8/bincount/Const_2?
8model_1/model/category_encoding_8/bincount/DenseBincountDenseBincount/model_1/model/string_lookup_8/Identity:output:06model_1/model/category_encoding_8/bincount/Minimum:z:0;model_1/model/category_encoding_8/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2:
8model_1/model/category_encoding_8/bincount/DenseBincount?
'model_1/model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_1/model/concatenate_1/concat/axis?
"model_1/model/concatenate_1/concatConcatV2'model_1/model/normalization/truediv:z:0?model_1/model/category_encoding/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_1/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_2/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_3/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_4/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_5/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_6/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_7/bincount/DenseBincount:output:0Amodel_1/model/category_encoding_8/bincount/DenseBincount:output:00model_1/model/concatenate_1/concat/axis:output:0*
N
*
T0*(
_output_shapes
:?????????? 2$
"model_1/model/concatenate_1/concat?
.model_1/sequential/dense/MatMul/ReadVariableOpReadVariableOp7model_1_sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype020
.model_1/sequential/dense/MatMul/ReadVariableOp?
model_1/sequential/dense/MatMulMatMul+model_1/model/concatenate_1/concat:output:06model_1/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2!
model_1/sequential/dense/MatMul?
/model_1/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp8model_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_1/sequential/dense/BiasAdd/ReadVariableOp?
 model_1/sequential/dense/BiasAddBiasAdd)model_1/sequential/dense/MatMul:product:07model_1/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2"
 model_1/sequential/dense/BiasAdd?
0model_1/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp9model_1_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype022
0model_1/sequential/dense_1/MatMul/ReadVariableOp?
!model_1/sequential/dense_1/MatMulMatMul)model_1/sequential/dense/BiasAdd:output:08model_1/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!model_1/sequential/dense_1/MatMul?
1model_1/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp:model_1_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_1/sequential/dense_1/BiasAdd/ReadVariableOp?
"model_1/sequential/dense_1/BiasAddBiasAdd+model_1/sequential/dense_1/MatMul:product:09model_1/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"model_1/sequential/dense_1/BiasAdd?
IdentityIdentity+model_1/sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

NoOpNoOp.^model_1/model/category_encoding/Assert/Assert0^model_1/model/category_encoding_1/Assert/Assert0^model_1/model/category_encoding_2/Assert/Assert0^model_1/model/category_encoding_3/Assert/Assert0^model_1/model/category_encoding_4/Assert/Assert0^model_1/model/category_encoding_5/Assert/Assert0^model_1/model/category_encoding_6/Assert/Assert0^model_1/model/category_encoding_7/Assert/Assert0^model_1/model/category_encoding_8/Assert/Assert:^model_1/model/string_lookup/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_1/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_2/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_3/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_4/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_5/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_6/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_7/None_Lookup/LookupTableFindV2<^model_1/model/string_lookup_8/None_Lookup/LookupTableFindV20^model_1/sequential/dense/BiasAdd/ReadVariableOp/^model_1/sequential/dense/MatMul/ReadVariableOp2^model_1/sequential/dense_1/BiasAdd/ReadVariableOp1^model_1/sequential/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : : : : : : : : : :	:	: : : : 2^
-model_1/model/category_encoding/Assert/Assert-model_1/model/category_encoding/Assert/Assert2b
/model_1/model/category_encoding_1/Assert/Assert/model_1/model/category_encoding_1/Assert/Assert2b
/model_1/model/category_encoding_2/Assert/Assert/model_1/model/category_encoding_2/Assert/Assert2b
/model_1/model/category_encoding_3/Assert/Assert/model_1/model/category_encoding_3/Assert/Assert2b
/model_1/model/category_encoding_4/Assert/Assert/model_1/model/category_encoding_4/Assert/Assert2b
/model_1/model/category_encoding_5/Assert/Assert/model_1/model/category_encoding_5/Assert/Assert2b
/model_1/model/category_encoding_6/Assert/Assert/model_1/model/category_encoding_6/Assert/Assert2b
/model_1/model/category_encoding_7/Assert/Assert/model_1/model/category_encoding_7/Assert/Assert2b
/model_1/model/category_encoding_8/Assert/Assert/model_1/model/category_encoding_8/Assert/Assert2v
9model_1/model/string_lookup/None_Lookup/LookupTableFindV29model_1/model/string_lookup/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_1/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_1/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_2/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_2/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_3/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_3/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_4/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_4/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_5/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_5/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_6/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_6/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_7/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_7/None_Lookup/LookupTableFindV22z
;model_1/model/string_lookup_8/None_Lookup/LookupTableFindV2;model_1/model/string_lookup_8/None_Lookup/LookupTableFindV22b
/model_1/sequential/dense/BiasAdd/ReadVariableOp/model_1/sequential/dense/BiasAdd/ReadVariableOp2`
.model_1/sequential/dense/MatMul/ReadVariableOp.model_1/sequential/dense/MatMul/ReadVariableOp2f
1model_1/sequential/dense_1/BiasAdd/ReadVariableOp1model_1/sequential/dense_1/BiasAdd/ReadVariableOp2d
0model_1/sequential/dense_1/MatMul/ReadVariableOp0model_1/sequential/dense_1/MatMul/ReadVariableOp:U Q
'
_output_shapes
:?????????
&
_user_specified_nameachievements:NJ
'
_output_shapes
:?????????

_user_specified_nameappid:YU
'
_output_shapes
:?????????
*
_user_specified_nameaverage_playtime:SO
'
_output_shapes
:?????????
$
_user_specified_name
categories:RN
'
_output_shapes
:?????????
#
_user_specified_name	developer:PL
'
_output_shapes
:?????????
!
_user_specified_name	english:OK
'
_output_shapes
:?????????
 
_user_specified_namegenres:XT
'
_output_shapes
:?????????
)
_user_specified_namemedian_playtime:MI
'
_output_shapes
:?????????

_user_specified_namename:Y	U
'
_output_shapes
:?????????
*
_user_specified_namenegative_ratings:O
K
'
_output_shapes
:?????????
 
_user_specified_nameowners:RN
'
_output_shapes
:?????????
#
_user_specified_name	platforms:YU
'
_output_shapes
:?????????
*
_user_specified_namepositive_ratings:NJ
'
_output_shapes
:?????????

_user_specified_nameprice:RN
'
_output_shapes
:?????????
#
_user_specified_name	publisher:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelease_date:UQ
'
_output_shapes
:?????????
&
_user_specified_namerequired_age:VR
'
_output_shapes
:?????????
'
_user_specified_namesteamspy_tags:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes
: :#

_output_shapes
: :$$ 

_output_shapes

:	:$% 

_output_shapes

:	"?N
saver_filename:0StatefulPartitionedCall_10:0StatefulPartitionedCall_118"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?

serving_default?

E
achievements5
serving_default_achievements:0?????????
7
appid.
serving_default_appid:0?????????
M
average_playtime9
"serving_default_average_playtime:0?????????
A

categories3
serving_default_categories:0?????????
?
	developer2
serving_default_developer:0?????????
;
english0
serving_default_english:0?????????
9
genres/
serving_default_genres:0?????????
K
median_playtime8
!serving_default_median_playtime:0?????????
5
name-
serving_default_name:0?????????
M
negative_ratings9
"serving_default_negative_ratings:0?????????
9
owners/
serving_default_owners:0?????????
?
	platforms2
serving_default_platforms:0?????????
M
positive_ratings9
"serving_default_positive_ratings:0?????????
7
price.
serving_default_price:0?????????
?
	publisher2
serving_default_publisher:0?????????
E
release_date5
serving_default_release_date:0?????????
E
required_age5
serving_default_required_age:0?????????
G
steamspy_tags6
serving_default_steamspy_tags:0?????????@

sequential2
StatefulPartitionedCall_9:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer_with_weights-0
layer-18
layer_with_weights-1
layer-19
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
layer-0
layer-1
layer-2
layer-3
layer-4

layer-5
layer-6
layer-7
layer-8
	layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
 layer-23
!layer-24
"layer-25
#layer-26
$layer-27
%layer_with_weights-0
%layer-28
&layer-29
'layer-30
(layer-31
)layer-32
*layer-33
+layer-34
,layer-35
-layer-36
.layer-37
/layer-38
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_network
?
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
:iter

;beta_1

<beta_2
	=decay
>learning_rateBm?Cm?Dm?Em?Bv?Cv?Dv?Ev?"
	optimizer
Q
?0
@1
A2
B3
C4
D5
E6"
trackable_list_wrapper
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
?
Fnon_trainable_variables
Glayer_metrics
	variables
regularization_losses
Hlayer_regularization_losses
Imetrics

Jlayers
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
:
Olookup_table
P	keras_api"
_tf_keras_layer
:
Qlookup_table
R	keras_api"
_tf_keras_layer
:
Slookup_table
T	keras_api"
_tf_keras_layer
:
Ulookup_table
V	keras_api"
_tf_keras_layer
:
Wlookup_table
X	keras_api"
_tf_keras_layer
:
Ylookup_table
Z	keras_api"
_tf_keras_layer
:
[lookup_table
\	keras_api"
_tf_keras_layer
:
]lookup_table
^	keras_api"
_tf_keras_layer
:
_lookup_table
`	keras_api"
_tf_keras_layer
?
a
_keep_axis
b_reduce_axis
c_reduce_axis_mask
d_broadcast_shape
?mean
?
adapt_mean
@variance
@adapt_variance
	Acount
e	keras_api
?_adapt_function"
_tf_keras_layer
?
f	variables
gregularization_losses
htrainable_variables
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
~	variables
regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
0	variables
1regularization_losses
 ?layer_regularization_losses
?metrics
?layers
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Bkernel
Cbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Dkernel
Ebias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
6	variables
7regularization_losses
 ?layer_regularization_losses
?metrics
?layers
8trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:	2mean
:	2variance
:	 2count
:	? @2dense/kernel
:@2
dense/bias
 :@2dense_1/kernel
:2dense_1/bias
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
K	variables
Lregularization_losses
 ?layer_regularization_losses
?metrics
?layers
Mtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
f	variables
gregularization_losses
 ?layer_regularization_losses
?metrics
?layers
htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
j	variables
kregularization_losses
 ?layer_regularization_losses
?metrics
?layers
ltrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
n	variables
oregularization_losses
 ?layer_regularization_losses
?metrics
?layers
ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
r	variables
sregularization_losses
 ?layer_regularization_losses
?metrics
?layers
ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
v	variables
wregularization_losses
 ?layer_regularization_losses
?metrics
?layers
xtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
z	variables
{regularization_losses
 ?layer_regularization_losses
?metrics
?layers
|trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
~	variables
regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4

5
6
7
8
	9
10
11
12
13
14
15
16
17
18
19
20
21
22
 23
!24
"25
#26
$27
%28
&29
'30
(31
)32
*33
+34
,35
-36
.37
/38"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
?non_trainable_variables
?layer_metrics
?	variables
?regularization_losses
 ?layer_regularization_losses
?metrics
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
$:"	? @2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
$:"	? @2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_11849
B__inference_model_1_layer_call_and_return_conditional_losses_12199
B__inference_model_1_layer_call_and_return_conditional_losses_11350
B__inference_model_1_layer_call_and_return_conditional_losses_11421?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_9724achievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tags"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_model_1_layer_call_fn_10998
'__inference_model_1_layer_call_fn_12269
'__inference_model_1_layer_call_fn_12339
'__inference_model_1_layer_call_fn_11279?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_model_layer_call_and_return_conditional_losses_12677
@__inference_model_layer_call_and_return_conditional_losses_13015
@__inference_model_layer_call_and_return_conditional_losses_10628
@__inference_model_layer_call_and_return_conditional_losses_10703?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_model_layer_call_fn_10209
%__inference_model_layer_call_fn_13077
%__inference_model_layer_call_fn_13139
%__inference_model_layer_call_fn_10553?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_13155
E__inference_sequential_layer_call_and_return_conditional_losses_13171
E__inference_sequential_layer_call_and_return_conditional_losses_10841
E__inference_sequential_layer_call_and_return_conditional_losses_10855?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_sequential_layer_call_fn_10754
*__inference_sequential_layer_call_fn_13184
*__inference_sequential_layer_call_fn_13197
*__inference_sequential_layer_call_fn_10827?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_11499achievementsappidaverage_playtime
categories	developerenglishgenresmedian_playtimenamenegative_ratingsowners	platformspositive_ratingsprice	publisherrelease_daterequired_agesteamspy_tags"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_13211?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_13224?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_13270?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_category_encoding_layer_call_and_return_conditional_losses_13304?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_category_encoding_layer_call_fn_13309?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_1_layer_call_and_return_conditional_losses_13343?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_1_layer_call_fn_13348?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_2_layer_call_and_return_conditional_losses_13382?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_2_layer_call_fn_13387?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_3_layer_call_and_return_conditional_losses_13421?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_3_layer_call_fn_13426?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_13460?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_4_layer_call_fn_13465?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_13499?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_5_layer_call_fn_13504?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_13538?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_6_layer_call_fn_13543?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_13577?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_7_layer_call_fn_13582?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_13616?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_category_encoding_8_layer_call_fn_13621?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_13636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_1_layer_call_fn_13650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_13660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_13669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_13679?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_13688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_13693?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13701?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13706?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13711?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13719?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13724?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13729?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13737?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13742?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13747?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13755?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13760?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13765?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13773?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13778?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13783?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13791?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13796?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13801?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13809?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13814?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13819?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13827?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13832?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_13837?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_13845?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_13850?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15
J

Const_16
J

Const_17
J

Const_18
J

Const_19
J

Const_20
J

Const_21
J

Const_22
J

Const_23
J

Const_24
J

Const_25
J

Const_26
J

Const_27
J

Const_286
__inference__creator_13693?

? 
? "? 6
__inference__creator_13711?

? 
? "? 6
__inference__creator_13729?

? 
? "? 6
__inference__creator_13747?

? 
? "? 6
__inference__creator_13765?

? 
? "? 6
__inference__creator_13783?

? 
? "? 6
__inference__creator_13801?

? 
? "? 6
__inference__creator_13819?

? 
? "? 6
__inference__creator_13837?

? 
? "? 8
__inference__destroyer_13706?

? 
? "? 8
__inference__destroyer_13724?

? 
? "? 8
__inference__destroyer_13742?

? 
? "? 8
__inference__destroyer_13760?

? 
? "? 8
__inference__destroyer_13778?

? 
? "? 8
__inference__destroyer_13796?

? 
? "? 8
__inference__destroyer_13814?

? 
? "? 8
__inference__destroyer_13832?

? 
? "? 8
__inference__destroyer_13850?

? 
? "? A
__inference__initializer_13701O???

? 
? "? A
__inference__initializer_13719Q???

? 
? "? A
__inference__initializer_13737S???

? 
? "? A
__inference__initializer_13755U???

? 
? "? A
__inference__initializer_13773W???

? 
? "? A
__inference__initializer_13791Y???

? 
? "? A
__inference__initializer_13809[???

? 
? "? A
__inference__initializer_13827]???

? 
? "? A
__inference__initializer_13845_???

? 
? "? ?
__inference__wrapped_model_9724?#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
? "7?4
2

sequential$?!

sequential?????????l
__inference_adapt_step_13270LA?@A?>
7?4
2?/?
??????????	IteratorSpec
? "
 ?
N__inference_category_encoding_1_layer_call_and_return_conditional_losses_13343]3?0
)?&
 ?
inputs?????????	

 
? "&?#
?
0??????????
? ?
3__inference_category_encoding_1_layer_call_fn_13348P3?0
)?&
 ?
inputs?????????	

 
? "????????????
N__inference_category_encoding_2_layer_call_and_return_conditional_losses_13382]3?0
)?&
 ?
inputs?????????	

 
? "&?#
?
0??????????
? ?
3__inference_category_encoding_2_layer_call_fn_13387P3?0
)?&
 ?
inputs?????????	

 
? "????????????
N__inference_category_encoding_3_layer_call_and_return_conditional_losses_13421]3?0
)?&
 ?
inputs?????????	

 
? "&?#
?
0??????????
? ?
3__inference_category_encoding_3_layer_call_fn_13426P3?0
)?&
 ?
inputs?????????	

 
? "????????????
N__inference_category_encoding_4_layer_call_and_return_conditional_losses_13460\3?0
)?&
 ?
inputs?????????	

 
? "%?"
?
0?????????
? ?
3__inference_category_encoding_4_layer_call_fn_13465O3?0
)?&
 ?
inputs?????????	

 
? "???????????
N__inference_category_encoding_5_layer_call_and_return_conditional_losses_13499]3?0
)?&
 ?
inputs?????????	

 
? "&?#
?
0??????????
? ?
3__inference_category_encoding_5_layer_call_fn_13504P3?0
)?&
 ?
inputs?????????	

 
? "????????????
N__inference_category_encoding_6_layer_call_and_return_conditional_losses_13538]3?0
)?&
 ?
inputs?????????	

 
? "&?#
?
0??????????
? ?
3__inference_category_encoding_6_layer_call_fn_13543P3?0
)?&
 ?
inputs?????????	

 
? "????????????
N__inference_category_encoding_7_layer_call_and_return_conditional_losses_13577]3?0
)?&
 ?
inputs?????????	

 
? "&?#
?
0??????????
? ?
3__inference_category_encoding_7_layer_call_fn_13582P3?0
)?&
 ?
inputs?????????	

 
? "????????????
N__inference_category_encoding_8_layer_call_and_return_conditional_losses_13616\3?0
)?&
 ?
inputs?????????	

 
? "%?"
?
0?????????
? ?
3__inference_category_encoding_8_layer_call_fn_13621O3?0
)?&
 ?
inputs?????????	

 
? "???????????
L__inference_category_encoding_layer_call_and_return_conditional_losses_13304]3?0
)?&
 ?
inputs?????????	

 
? "&?#
?
0??????????
? ?
1__inference_category_encoding_layer_call_fn_13309P3?0
)?&
 ?
inputs?????????	

 
? "????????????
H__inference_concatenate_1_layer_call_and_return_conditional_losses_13636????
???
???
"?
inputs/0?????????	
#? 
inputs/1??????????
#? 
inputs/2??????????
#? 
inputs/3??????????
#? 
inputs/4??????????
"?
inputs/5?????????
#? 
inputs/6??????????
#? 
inputs/7??????????
#? 
inputs/8??????????
"?
inputs/9?????????
? "&?#
?
0?????????? 
? ?
-__inference_concatenate_1_layer_call_fn_13650????
???
???
"?
inputs/0?????????	
#? 
inputs/1??????????
#? 
inputs/2??????????
#? 
inputs/3??????????
#? 
inputs/4??????????
"?
inputs/5?????????
#? 
inputs/6??????????
#? 
inputs/7??????????
#? 
inputs/8??????????
"?
inputs/9?????????
? "??????????? ?
F__inference_concatenate_layer_call_and_return_conditional_losses_13211????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
? "%?"
?
0?????????	
? ?
+__inference_concatenate_layer_call_fn_13224????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????
? "??????????	?
B__inference_dense_1_layer_call_and_return_conditional_losses_13679\DE/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_13688ODE/?,
%?"
 ?
inputs?????????@
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_13660]BC0?-
&?#
!?
inputs?????????? 
? "%?"
?
0?????????@
? y
%__inference_dense_layer_call_fn_13669PBC0?-
&?#
!?
inputs?????????? 
? "??????????@?
B__inference_model_1_layer_call_and_return_conditional_losses_11350?#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_11421?#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p

 
? "%?"
?
0?????????
? ?	
B__inference_model_1_layer_call_and_return_conditional_losses_11849?	#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p 

 
? "%?"
?
0?????????
? ?	
B__inference_model_1_layer_call_and_return_conditional_losses_12199?	#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p

 
? "%?"
?
0?????????
? ?
'__inference_model_1_layer_call_fn_10998?#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p 

 
? "???????????
'__inference_model_1_layer_call_fn_11279?#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p

 
? "???????????	
'__inference_model_1_layer_call_fn_12269?	#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p 

 
? "???????????	
'__inference_model_1_layer_call_fn_12339?	#_?]?[?Y?W?U?S?Q?O???BCDE???
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p

 
? "???????????
@__inference_model_layer_call_and_return_conditional_losses_10628?_?]?[?Y?W?U?S?Q?O??????
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p 

 
? "&?#
?
0?????????? 
? ?
@__inference_model_layer_call_and_return_conditional_losses_10703?_?]?[?Y?W?U?S?Q?O??????
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p

 
? "&?#
?
0?????????? 
? ?	
@__inference_model_layer_call_and_return_conditional_losses_12677?	_?]?[?Y?W?U?S?Q?O??????
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p 

 
? "&?#
?
0?????????? 
? ?	
@__inference_model_layer_call_and_return_conditional_losses_13015?	_?]?[?Y?W?U?S?Q?O??????
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p

 
? "&?#
?
0?????????? 
? ?
%__inference_model_layer_call_fn_10209?_?]?[?Y?W?U?S?Q?O??????
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p 

 
? "??????????? ?
%__inference_model_layer_call_fn_10553?_?]?[?Y?W?U?S?Q?O??????
???
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????
p

 
? "??????????? ?	
%__inference_model_layer_call_fn_13077?	_?]?[?Y?W?U?S?Q?O??????
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p 

 
? "??????????? ?	
%__inference_model_layer_call_fn_13139?	_?]?[?Y?W?U?S?Q?O??????
???
???
=
achievements-?*
inputs/achievements?????????
/
appid&?#
inputs/appid?????????
E
average_playtime1?.
inputs/average_playtime?????????
9

categories+?(
inputs/categories?????????
7
	developer*?'
inputs/developer?????????
3
english(?%
inputs/english?????????
1
genres'?$
inputs/genres?????????
C
median_playtime0?-
inputs/median_playtime?????????
-
name%?"
inputs/name?????????
E
negative_ratings1?.
inputs/negative_ratings?????????
1
owners'?$
inputs/owners?????????
7
	platforms*?'
inputs/platforms?????????
E
positive_ratings1?.
inputs/positive_ratings?????????
/
price&?#
inputs/price?????????
7
	publisher*?'
inputs/publisher?????????
=
release_date-?*
inputs/release_date?????????
=
required_age-?*
inputs/required_age?????????
?
steamspy_tags.?+
inputs/steamspy_tags?????????
p

 
? "??????????? ?
E__inference_sequential_layer_call_and_return_conditional_losses_10841lBCDE=?:
3?0
&?#
dense_input?????????? 
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_10855lBCDE=?:
3?0
&?#
dense_input?????????? 
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_13155gBCDE8?5
.?+
!?
inputs?????????? 
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_13171gBCDE8?5
.?+
!?
inputs?????????? 
p

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_10754_BCDE=?:
3?0
&?#
dense_input?????????? 
p 

 
? "???????????
*__inference_sequential_layer_call_fn_10827_BCDE=?:
3?0
&?#
dense_input?????????? 
p

 
? "???????????
*__inference_sequential_layer_call_fn_13184ZBCDE8?5
.?+
!?
inputs?????????? 
p 

 
? "???????????
*__inference_sequential_layer_call_fn_13197ZBCDE8?5
.?+
!?
inputs?????????? 
p

 
? "???????????
#__inference_signature_wrapper_11499?#_?]?[?Y?W?U?S?Q?O???BCDE???
? 
???
6
achievements&?#
achievements?????????
(
appid?
appid?????????
>
average_playtime*?'
average_playtime?????????
2

categories$?!

categories?????????
0
	developer#? 
	developer?????????
,
english!?
english?????????
*
genres ?
genres?????????
<
median_playtime)?&
median_playtime?????????
&
name?
name?????????
>
negative_ratings*?'
negative_ratings?????????
*
owners ?
owners?????????
0
	platforms#? 
	platforms?????????
>
positive_ratings*?'
positive_ratings?????????
(
price?
price?????????
0
	publisher#? 
	publisher?????????
6
release_date&?#
release_date?????????
6
required_age&?#
required_age?????????
8
steamspy_tags'?$
steamspy_tags?????????"7?4
2

sequential$?!

sequential?????????