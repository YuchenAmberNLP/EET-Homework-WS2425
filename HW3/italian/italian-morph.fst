%%% Single Character Symbols %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#letter# = a-zA-ZàèéòìùÀÈÉÒÌÙ

%%% Analysis Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Adjectives ending with -e have only two forms, varying only according to number (singular/plural).
% number feature
#Number# = <sg><pl>

% gender feature
#Gender# = <masc><fem>

% Person Feature
#Person# = <1><2><3>

%%% Agreement Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#WordArt# = <ADJ><VERB>

% verbal features
%%% There are 3 moods (Indicative Mood, Conditional mood, Subjunctive mood, Imperative mood) for verbs in Italian, for Indicative Mood, there are 4 tenses (present, imperfect, preterite, and future), for conditional mood, there is only 1 tense (present), for subjunctive mood, there are 2 tenses(present), and for imperative mood, there is only present tense conjunction.
#VerbTense# = <present><imperfect><preterite><future>
#VerbMood# = <indicative><conditional><subjunctive><imperative>
#VerbNominalForm# = <infinite><participle><gerund>

% Morphosyntactic Features
#MorphSyn# = #Number# #Gender# #Person# #VerbTense# #VerbMood# #VerbNominalForm#

ALPHABET = [#letter#]

%%% definition of the inflectional classes %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%-ADJEKTIVE-Regeln%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% $ADJ-os$: Bearbeitung von Adjektiven, die Maskuline Singular From auf -o enden
%%% Beispiel: "buono" -> "buon"
$ADJ-os$ = "adjectives-o.lex" || [a-z]* {o}:{} <ADJ>:<>

%%% $ADJ-es$: Bearbeitung von Adjektiven, die Maskuline Singular From auf -e enden
%%% Beispiel: "felice" -> "felic"
$ADJ-es$ = "adjectives-e.lex" || [a-z]* {e}:{} <ADJ>:<>

%%% $AdjReg-o$: Diese Ersetzungsregel behandelt Adjektive, die im Maskulinum Singular auf "-o" enden. Der Prozess des Entfernens des letzten "o" wurde bereits in der Regel $ADJ-os$ beschrieben, bei der das "-o" von den Adjektivstämmen entfernt wird. Diese Regel fügt die entsprechende Endung basierend auf Geschlecht (maskulin, feminin) und Anzahl (Singular, Plural) hinzu.
%%% Beispiel:
%%%  - Maskulinum Singular: {<masc><sg>} "buon" + "o" -> "buono" (das "o" wird für den Singular Maskulinum hinzugefügt)
%%%  - Maskulinum Plural: {<masc><pl>} "buon" + "i" -> "buoni" (das "i" ersetzt das "o" für den Maskulinum Plural)
%%%  - Femininum Singular: {<fem><sg>} "buon" + "a" -> "buona" (das "a" ersetzt das "o" für das Femininum Singular)
%%%  - Femininum Plural: {<fem><pl>} "buon" + "e" -> "buone" (das "e" ersetzt das "o" für den Femininum Plural)

$AdjReg-o$ = {<masc><sg>}:{o} |\
             {<masc><pl>}:{i} |\
             {<fem><sg>}:{a} |\
             {<fem><pl>}:{e}
%%% Diese Regel ergänzt die in $ADJ-os$ beschriebene Entfernung des letzten "o" und ersetzt es durch die richtige Endung basierend auf den Flexionen des Adjektivs.

%%% $AdjReg-e$: Diese Ersetzungsregel behandelt Adjektive, die sowohl im Maskulinum als auch im Femininum Singular auf "-e" enden. Der Prozess des Entfernens des letzten "e" wurde bereits in der Regel $ADJ-es$ beschrieben, bei der das "-e" von den Adjektivstämmen entfernt wird. Diese Regel fügt die entsprechende Endung basierend auf Geschlecht (maskulin, feminin) und Anzahl (Singular, Plural) hinzu.
%%% Beispiel:
%%%  - Maskulinum Singular: {<masc><sg>} "fort" + "e" -> "forte" (das "e" wird für den Singular beider Geschlechter hinzugefügt)
%%%  - Maskulinum Plural: {<masc><pl>} "fort" + "i" -> "forti" (das "i" ersetzt das "e" für den Maskulinum Plural)
%%%  - Femininum Singular: {<fem><sg>} "fort" + "e" -> "forte" (das "e" bleibt für das Femininum Singular unverändert)
%%%  - Femininum Plural: {<fem><pl>} "fort" + "i" -> "forti" (das "i" ersetzt das "e" für den Femininum Plural)

$AdjReg-e$ = {<masc><sg>}:{e} |\
             {<masc><pl>}:{i} |\
             {<fem><sg>}:{e} |\
             {<fem><pl>}:{i}
%%% Diese Regel ergänzt die in $ADJ-es$ beschriebene Entfernung des letzten "e" und ersetzt es durch die passende Endung basierend auf den Flexionen des Adjektivs.

%%%%%%%%%-VERB-Regeln%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% $verbstems-are$ Bearbeitung von Verben, die 1st Conjugation Form auf -are enden
%%% Beispiel: "amare" -> "am"

$verbstems-are$ = "verbs-are.lex" || [a-z]* {are}:{} <VERB>:<>

%%% $VerbRegInd-are$: Diese Ersetzungsregel gilt für Verben, die je nach Zeitform im Indikativ, Konjunktiv, Konditional oder Imperativ auf -are enden.
%%% Beispiel:
%%%  - Indikativ Präsens 1st Person Singular: {<indicative><present><1><sg>} "am" + "o" -> "amo" (das "o" ersetzt das "are")
%%%  - Indikativ Präsens 2te Person Singular: {<indicative><present><2><sg>} "am" + "i" -> "ami" (das "i" ersetzt das "are")
%%%  - Indikativ Präsens 3te Person Singular: {<indicative><present><3><sg>} "am" + "a" -> "ama" (das "a" ersetzt das "are")
%%%  - Indikativ Präsens 1st Person Plural: {<indicative><present><1><pl>} "am" + "iamo" -> "amiamo" (das "iamo" ersetzt das "are")
%%%  - Indikativ Präsens 2te Person Plural: {<indicative><present><2><pl>} "am" + "ate" -> "amate" (das "ate" ersetzt das "are")
%%%  - Indikativ Präsens 3te Person Plural: {<indicative><present><3><pl>} "am" + "ano" -> "amano" (das "ano" ersetzt das "are")
%%%  - Indikativ Imperfekt 1st Person Singular: {<indicative><imperfect><1><sg>} "am" + "avo" -> "amavo" (das "avo" ersetzt das "are")
%%%  - Indikativ Imperfekt 2te Person Singular: {<indicative><imperfect><2><sg>} "am" + "avi" -> "amavi" (das "avi" ersetzt das "are")
%%%  - Indikativ Imperfekt 3te Person Singular: {<indicative><imperfect><3><sg>} "am" + "ava" -> "amava" (das "ava" ersetzt das "are")
%%%  - Indikativ Imperfekt 1st Person Plural: {<indicative><imperfect><1><pl>} "am" + "avamo" -> "amavamo" (das "avamo" ersetzt das "are")
%%%  - Indikativ Imperfekt 2te Person Plural: {<indicative><imperfect><2><pl>} "am" + "avate" -> "amavate" (das "avate" ersetzt das "are")
%%%  - Indikativ Imperfekt 3te Person Plural: {<indicative><imperfect><3><pl>} "am" + "avano" -> "amavano" (das "avano" ersetzt das "are")
%%%  - Indikativ Praeteritum 1st Person Singular: {<indicative><preterite><1><sg>} "am" + "ai" -> "amai" (das "ai" ersetzt das "are")
%%%  - Indikativ Praeteritum 2te Person Singular: {<indicative><preterite><2><sg>} "am" + "asti" -> "amasti" (das "asti" ersetzt das "are")
%%%  - Indikativ Praeteritum 3te Person Singular: {<indicative><preterite><3><sg>} "am" + "ò" -> "amò" (das "ò" ersetzt das "are")
%%%  - Indikativ Praeteritum 1st Person Plural: {<indicative><preterite><1><pl>} "am" + "ammo" -> "amammo" (das "ammo" ersetzt das "are")
%%%  - Indikativ Praeteritum 2te Person Plural: {<indicative><preterite><2><pl>} "am" + "aste" -> "amaste" (das "aste" ersetzt das "are")
%%%  - Indikativ Praeteritum 3te Person Plural: {<indicative><preterite><3><pl>} "am" + "arono" -> "amarono" (das "arono" ersetzt das "are")
%%%  - Indikativ Future 1st Person Singular: {<indicative><future><1><sg>} "am" + "erò" -> "amerò" (das "erò" ersetzt das "are")
%%%  - Indikativ Future 2te Person Singular: {<indicative><future><2><sg>} "am" + "erai" -> "amerai" (das "erai" ersetzt das "are")
%%%  - Indikativ Future 3te Person Singular: {<indicative><future><3><sg>} "am" + "erà" -> "amerà" (das "erà" ersetzt das "are")
%%%  - Indikativ Future 1st Person Plural: {<indicative><future><1><pl>} "am" + "eremo" -> "ameremo" (das "eremo" ersetzt das "are")
%%%  - Indikativ Future 2te Person Plural: {<indicative><future><2><pl>} "am" + "erete" -> "amerete" (das "erete" ersetzt das "are")
%%%  - Indikativ Future 3te Person Plural: {<indicative><future><3><pl>} "am" + "eranno" -> "ameranno" (das "eranno" ersetzt das "are")

%%%  Auch die Zeitformen von Konditional, Konjunktiv, Subjunktiv und Imperativ sind ähnlich.

$VerbRegInd-are$ = {<indicative><present><1><sg>}:{o} |\
                {<indicative><present><2><sg>}:{i} |\
                {<indicative><present><3><sg>}:{a} |\
                {<indicative><present><1><pl>}:{iamo} |\
                {<indicative><present><2><pl>}:{ate} |\
                {<indicative><present><3><pl>}:{ano} |\
                {<indicative><imperfect><1><sg>}:{avo} |\
                {<indicative><imperfect><2><sg>}:{avi} |\
                {<indicative><imperfect><3><sg>}:{ava} |\
                {<indicative><imperfect><1><pl>}:{avamo} |\
                {<indicative><imperfect><2><pl>}:{avate} |\
                {<indicative><imperfect><3><pl>}:{avano} |\
                {<indicative><preterite><1><sg>}:{ai} |\
                {<indicative><preterite><2><sg>}:{asti} |\
                {<indicative><preterite><3><sg>}:{ò} |\
                {<indicative><preterite><1><pl>}:{ammo} |\
                {<indicative><preterite><2><pl>}:{aste} |\
                {<indicative><preterite><3><pl>}:{arono} |\
                {<indicative><future><1><sg>}:{erò} |\
                {<indicative><future><2><sg>}:{erai} |\
                {<indicative><future><3><sg>}:{erà} |\
                {<indicative><future><1><pl>}:{eremo} |\
                {<indicative><future><2><pl>}:{erete} |\
                {<indicative><future><3><pl>}:{eranno}


%%% $VerbRegCond-are$: Diese Ersetzungsregel beschreibt die Konjugation von Verben, die im Konditional Präsens auf -are enden, nach Person und Anzahl.
%%% Beispiel:
%%%  - Konditional Präsens 1st Person Singular: {<conditional><present><1><sg>} "am" + "erei" -> "amerei" (das "erei" ersetzt das "are")
%%%  - Konditional Präsens 2te Person Singular: {<conditional><present><2><sg>} "am" + "eresti" -> "ameresti" (das "eresti" ersetzt das "are")
%%%  - Konditional Präsens 3te Person Singular: {<conditional><present><3><sg>} "am" + "erebbe" -> "amerebbe" (das "erebbe" ersetzt das "are")
%%%  - Konditional Präsens 1st Person Plural: {<conditional><present><1><pl>} "am" + "eremmo" -> "ameremmo" (das "eremmo" ersetzt das "are")
%%%  - Konditional Präsens 2te Person Plural: {<conditional><present><2><pl>} "am" + "ereste" -> "amereste" (das "ereste" ersetzt das "are")
%%%  - Konditional Präsens 3te Person Plural: {<conditional><present><3><pl>} "am" + "erebbero" -> "amerebbero" (das "erebbero" ersetzt das "are")

$VerbRegCond-are$ = {<conditional><present><1><sg>}:{erei} |\
                {<conditional><present><2><sg>}:{eresti} |\
                {<conditional><present><3><sg>}:{erebbe} |\
                {<conditional><present><1><pl>}:{eremmo} |\
                {<conditional><present><2><pl>}:{ereste} |\
                {<conditional><present><3><pl>}:{erebbero}


%%% $VerbRegSubj-are$: Diese Ersetzungsregel beschreibt die Konjugation von Verben im Konjunktiv Präsens und Imperfekt für Verben, die auf -are enden.
%%% Beispiel:
%%%  - Konjunktiv Präsens 1st Person Singular: {<subjunctive><present><1><sg>} "am" + "i" -> "ami" (das "i" ersetzt das "are")
%%%  - Konjunktiv Präsens 2te Person Singular: {<subjunctive><present><2><sg>} "am" + "i" -> "ami" (das "i" ersetzt das "are")
%%%  - Konjunktiv Präsens 3te Person Singular: {<subjunctive><present><3><sg>} "am" + "i" -> "ami" (das "i" ersetzt das "are")
%%%  - Konjunktiv Präsens 1st Person Plural: {<subjunctive><present><1><pl>} "am" + "iamo" -> "amiamo" (das "iamo" ersetzt das "are")
%%%  - Konjunktiv Präsens 2te Person Plural: {<subjunctive><present><2><pl>} "am" + "iate" -> "amiate" (das "iate" ersetzt das "are")
%%%  - Konjunktiv Präsens 3te Person Plural: {<subjunctive><present><3><pl>} "am" + "ino" -> "amino" (das "ino" ersetzt das "are")
%%%  - Konjunktiv Imperfekt 1st Person Singular: {<subjunctive><imperfect><1><sg>} "am" + "assi" -> "amassi" (das "assi" ersetzt das "are")
%%%  - Konjunktiv Imperfekt 2te Person Singular: {<subjunctive><imperfect><2><sg>} "am" + "assi" -> "amassi" (das "assi" ersetzt das "are")
%%%  - Konjunktiv Imperfekt 3te Person Singular: {<subjunctive><imperfect><3><sg>} "am" + "asse" -> "amasse" (das "asse" ersetzt das "are")
%%%  - Konjunktiv Imperfekt 1st Person Plural: {<subjunctive><imperfect><1><pl>} "am" + "assimo" -> "amassimo" (das "assimo" ersetzt das "are")
%%%  - Konjunktiv Imperfekt 2te Person Plural: {<subjunctive><imperfect><2><pl>} "am" + "aste" -> "amaste" (das "aste" ersetzt das "are")
%%%  - Konjunktiv Imperfekt 3te Person Plural: {<subjunctive><imperfect><3><pl>} "am" + "assero" -> "amassero" (das "assero" ersetzt das "are")

$VerbRegSubj-are$ = {<subjunctive><present><1><sg>}:{i} |\
                {<subjunctive><present><2><sg>}:{i} |\
                {<subjunctive><present><3><sg>}:{i} |\
                {<subjunctive><present><1><pl>}:{iamo} |\
                {<subjunctive><present><2><pl>}:{iate} |\
                {<subjunctive><present><3><pl>}:{ino} |\
                {<subjunctive><imperfect><1><sg>}:{assi} |\
                {<subjunctive><imperfect><2><sg>}:{assi} |\
                {<subjunctive><imperfect><3><sg>}:{asse} |\
                {<subjunctive><imperfect><1><pl>}:{assimo} |\
                {<subjunctive><imperfect><2><pl>}:{aste} |\
                {<subjunctive><imperfect><3><pl>}:{assero}


%%% $VerbRegCondImp-are$: Diese Ersetzungsregel beschreibt die Bildung des Imperativs für Verben, die auf -are enden.
%%% Beispiel:
%%%  - Imperativ 2te Person Singular: {<imperative><present><2><sg>} "am" + "a" -> "ama" (das "a" ersetzt das "are")
%%%  - Imperativ 3te Person Singular: {<imperative><present><3><sg>} "am" + "i" -> "ami" (das "i" ersetzt das "are")
%%%  - Imperativ 1st Person Plural: {<imperative><present><1><pl>} "am" + "iamo" -> "amiamo" (das "iamo" ersetzt das "are")
%%%  - Imperativ 2te Person Plural: {<imperative><present><2><pl>} "am" + "ate" -> "amate" (das "ate" ersetzt das "are")
%%%  - Imperativ 3te Person Plural: {<imperative><present><3><pl>} "am" + "ino" -> "amino" (das "ino" ersetzt das "are")

$VerbRegCondImp-are$ = {<imperative><present><2><sg>}:{a} |\
                {<imperative><present><3><sg>}:{i} |\
                {<imperative><present><1><pl>}:{iamo} |\
                {<imperative><present><2><pl>}:{ate} |\
                {<imperative><present><3><pl>}:{ino}


%%% $VerbRegInf-are$: Diese Ersetzungsregel beschreibt die Bildung des Infinitivs Präsens für Verben, die auf -are enden.
%%% Beispiel:
%%%  - Infinitiv Präsens: {<infinite><present>} "am" + "are" -> "amare" (das "are" bleibt bestehen)

$VerbRegInf-are$ = {<infinite><present>}:{are}


%%% $VerbRegGerund-are$: Diese Ersetzungsregel beschreibt die Bildung des Gerundiums für Verben, die auf -are enden.
%%% Beispiel:
%%%  - Gerundium Präsens: {<gerund><present>} "am" + "ando" -> "amando" (das "ando" ersetzt das "are")
%%%  - Gerundium Perfekt: {<gerund><preterite>} "am" + "ato" -> "amato" (das "ato" ersetzt das "are")

$VerbRegGerund-are$ = {<gerund><present>}:{ando} |\
                {<gerund><preterite>}:{ato}


%%% $VerbRegPart-are$: Diese Ersetzungsregel beschreibt die Bildung des Partizips für Verben, die auf -are enden.
%%% Beispiel:
%%%  - Partizip Präsens: {<participle><present>} "am" + "ante" -> "amante" (das "ante" ersetzt das "are")
%%%  - Partizip Perfekt: {<participle><preterite>} "am" + "ato" -> "amato" (das "ato" ersetzt das "are")

$VerbRegPart-are$ = {<participle><present>}:{ante} |\
                {<participle><preterite>}:{ato}

%%% Die obigen Regeln ergänzen die in $verbstems-are$ beschriebene Entfernung des letzten "are" und ersetzen es durch die passende Endung, die auf den Flexionen des Verbs basiert.

%%% Vereinigen Sie die Transducer fuer die verschiedenen Flexionsklassen mit dem Disjunktions-Operator

($ADJ-es$ $AdjReg-e$) | ($ADJ-os$ $AdjReg-o$) | ($verbstems-are$ ($VerbRegInd-are$ | $VerbRegCond-are$ | $VerbRegSubj-are$ | $VerbRegCondImp-are$ | $VerbRegInf-are$ | $VerbRegGerund-are$ | $VerbRegPart-are$))
