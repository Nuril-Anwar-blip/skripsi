// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FederatedLearningBlockchain
 * @dev Smart contract untuk mengatur Federated Learning dengan keamanan blockchain
 * 
 * Features:
 * - Mencatat semua model updates dari hospitals/clients
 * - Reward system untuk honest participants
 * - Reputation management
 * - Byzantine detection
 * - Immutable audit trail
 */

contract FederatedLearningBlockchain {
    
    // ========================================================================
    // STRUCTS & ENUMS
    // ========================================================================
    
    struct Hospital {
        address walletAddress;
        string hospitalName;
        uint256 registrationTime;
        uint256 reputation;  // 0-100 score
        uint256 updates_submitted;
        bool isActive;
    }
    
    struct ModelUpdate {
        uint256 round;
        address hospital;
        string modelHash;  // SHA-256 hash dari model weights
        uint256 timestamp;
        bool verified;
        uint256 accuracy;  // accuracy in basis points (0-10000 = 0-100%)
        string updateDetails;  // JSON string with update info
    }
    
    struct Round {
        uint256 roundNumber;
        uint256 startTime;
        uint256 endTime;
        uint256 num_hospitals;
        string globalModelHash;
        bool completed;
        mapping(address => string) hospitalUpdates;
    }
    
    enum UpdateStatus {
        PENDING,
        VERIFIED,
        REJECTED,
        SUSPICIOUS
    }
    
    // ========================================================================
    // STATE VARIABLES
    // ========================================================================
    
    address public owner;
    uint256 public constant REWARD_PER_UPDATE = 100;  // tokens
    uint256 public constant REPUTATION_INCREASE = 5;   // per honest update
    uint256 public constant REPUTATION_PENALTY = -20;  // per malicious update
    
    mapping(address => Hospital) public hospitals;
    mapping(uint256 => ModelUpdate[]) public roundUpdates;  // round -> updates
    mapping(uint256 => Round) public rounds;
    
    address[] public registeredHospitals;
    uint256 public currentRound = 0;
    uint256 public totalRewardsDistributed = 0;
    
    // For Byzantine detection
    mapping(uint256 => mapping(address => bool)) public suspiciousUpdates;
    
    // ========================================================================
    // EVENTS
    // ========================================================================
    
    event HospitalRegistered(
        address indexed hospital,
        string name,
        uint256 timestamp
    );
    
    event ModelUpdateSubmitted(
        uint256 indexed round,
        address indexed hospital,
        string modelHash,
        uint256 accuracy
    );
    
    event UpdateVerified(
        uint256 indexed round,
        address indexed hospital,
        string modelHash,
        bool isValid
    );
    
    event ReputationChanged(
        address indexed hospital,
        uint256 oldReputation,
        uint256 newReputation,
        string reason
    );
    
    event RewardDistributed(
        address indexed hospital,
        uint256 amount,
        string reason
    );
    
    event ByzantineDetected(
        uint256 indexed round,
        address indexed hospital,
        string reason
    );
    
    event RoundCompleted(
        uint256 indexed round,
        uint256 hospitalCount,
        string globalModelHash
    );
    
    // ========================================================================
    // MODIFIERS
    // ========================================================================
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }
    
    modifier onlyRegisteredHospital() {
        require(hospitals[msg.sender].isActive, "Hospital not registered");
        _;
    }
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    constructor() {
        owner = msg.sender;
        currentRound = 1;
    }
    
    // ========================================================================
    // HOSPITAL MANAGEMENT
    // ========================================================================
    
    /**
     * @dev Register a new hospital in the FL network
     */
    function registerHospital(
        address hospitalAddress,
        string memory hospitalName
    ) public onlyOwner {
        require(
            hospitals[hospitalAddress].registrationTime == 0,
            "Hospital already registered"
        );
        require(bytes(hospitalName).length > 0, "Invalid hospital name");
        
        hospitals[hospitalAddress] = Hospital({
            walletAddress: hospitalAddress,
            hospitalName: hospitalName,
            registrationTime: block.timestamp,
            reputation: 50,  // Start with neutral reputation
            updates_submitted: 0,
            isActive: true
        });
        
        registeredHospitals.push(hospitalAddress);
        
        emit HospitalRegistered(hospitalAddress, hospitalName, block.timestamp);
    }
    
    /**
     * @dev Get hospital info
     */
    function getHospitalInfo(address hospitalAddress)
        public
        view
        returns (Hospital memory)
    {
        return hospitals[hospitalAddress];
    }
    
    /**
     * @dev Get all registered hospitals
     */
    function getRegisteredHospitals()
        public
        view
        returns (address[] memory)
    {
        return registeredHospitals;
    }
    
    // ========================================================================
    // MODEL UPDATE SUBMISSION
    // ========================================================================
    
    /**
     * @dev Submit model update from hospital
     */
    function submitModelUpdate(
        uint256 round,
        string memory modelHash,
        uint256 accuracy,
        string memory updateDetails
    ) public onlyRegisteredHospital {
        require(round == currentRound, "Invalid round number");
        require(bytes(modelHash).length == 64, "Invalid model hash");
        require(accuracy <= 10000, "Invalid accuracy value");
        
        ModelUpdate memory update = ModelUpdate({
            round: round,
            hospital: msg.sender,
            modelHash: modelHash,
            timestamp: block.timestamp,
            verified: false,
            accuracy: accuracy,
            updateDetails: updateDetails
        });
        
        roundUpdates[round].push(update);
        hospitals[msg.sender].updates_submitted += 1;
        
        emit ModelUpdateSubmitted(
            round,
            msg.sender,
            modelHash,
            accuracy
        );
    }
    
    /**
     * @dev Get all updates for a round
     */
    function getRoundUpdates(uint256 round)
        public
        view
        returns (ModelUpdate[] memory)
    {
        return roundUpdates[round];
    }
    
    // ========================================================================
    // VERIFICATION & BYZANTINE DETECTION
    // ========================================================================
    
    /**
     * @dev Verify model update
     * Checks if the update is valid based on model hash and accuracy
     */
    function verifyModelUpdate(
        uint256 round,
        address hospital,
        string memory modelHash,
        bool isValid
    ) public onlyOwner {
        ModelUpdate[] storage updates = roundUpdates[round];
        
        for (uint256 i = 0; i < updates.length; i++) {
            if (updates[i].hospital == hospital &&
                keccak256(abi.encodePacked(updates[i].modelHash)) ==
                keccak256(abi.encodePacked(modelHash))) {
                
                updates[i].verified = isValid;
                
                if (isValid) {
                    // Reward honest behavior
                    rewardHospital(hospital, "Honest model update");
                    _increaseReputation(hospital, REPUTATION_INCREASE);
                } else {
                    // Penalize suspicious behavior
                    _decreaseReputation(hospital, REPUTATION_PENALTY);
                    suspiciousUpdates[round][hospital] = true;
                    emit ByzantineDetected(round, hospital, "Invalid model hash");
                }
                
                emit UpdateVerified(round, hospital, modelHash, isValid);
                return;
            }
        }
        
        revert("Update not found");
    }
    
    /**
     * @dev Detect Byzantine (malicious) updates
     * Uses distance-based detection: if update differs too much from others
     */
    function detectByzantineUpdate(
        uint256 round,
        address hospital,
        uint256 meanAccuracy,
        uint256 threshold  // In basis points
    ) public onlyOwner {
        ModelUpdate[] storage updates = roundUpdates[round];
        
        for (uint256 i = 0; i < updates.length; i++) {
            if (updates[i].hospital == hospital) {
                uint256 diff = updates[i].accuracy > meanAccuracy
                    ? updates[i].accuracy - meanAccuracy
                    : meanAccuracy - updates[i].accuracy;
                
                if (diff > threshold) {
                    suspiciousUpdates[round][hospital] = true;
                    _decreaseReputation(hospital, REPUTATION_PENALTY);
                    emit ByzantineDetected(
                        round,
                        hospital,
                        "Accuracy deviation detected"
                    );
                    return;
                }
            }
        }
    }
    
    /**
     * @dev Check if hospital update is suspicious
     */
    function isSuspiciousUpdate(uint256 round, address hospital)
        public
        view
        returns (bool)
    {
        return suspiciousUpdates[round][hospital];
    }
    
    // ========================================================================
    // REPUTATION MANAGEMENT
    // ========================================================================
    
    /**
     * @dev Increase hospital reputation
     */
    function _increaseReputation(address hospital, uint256 amount) internal {
        uint256 oldReputation = hospitals[hospital].reputation;
        uint256 newReputation = oldReputation + amount;
        
        if (newReputation > 100) {
            newReputation = 100;
        }
        
        hospitals[hospital].reputation = newReputation;
        
        emit ReputationChanged(
            hospital,
            oldReputation,
            newReputation,
            "Reputation increased for honest behavior"
        );
    }
    
    /**
     * @dev Decrease hospital reputation
     */
    function _decreaseReputation(address hospital, uint256 amount) internal {
        uint256 oldReputation = hospitals[hospital].reputation;
        int256 newReputation = int256(oldReputation) - int256(amount);
        
        if (newReputation < 0) {
            newReputation = 0;
        }
        
        hospitals[hospital].reputation = uint256(newReputation);
        
        // Deactivate if reputation too low
        if (newReputation < 20) {
            hospitals[hospital].isActive = false;
        }
        
        emit ReputationChanged(
            hospital,
            oldReputation,
            uint256(newReputation),
            "Reputation decreased for suspicious behavior"
        );
    }
    
    /**
     * @dev Get hospital reputation
     */
    function getReputation(address hospital)
        public
        view
        returns (uint256)
    {
        return hospitals[hospital].reputation;
    }
    
    // ========================================================================
    // REWARDS & INCENTIVES
    // ========================================================================
    
    /**
     * @dev Reward hospital for contribution
     */
    function rewardHospital(address hospital, string memory reason) public onlyOwner {
        require(hospitals[hospital].isActive, "Hospital not active");
        
        uint256 rewardAmount = REWARD_PER_UPDATE;
        totalRewardsDistributed += rewardAmount;
        
        emit RewardDistributed(hospital, rewardAmount, reason);
    }
    
    /**
     * @dev Get total rewards distributed
     */
    function getTotalRewards() public view returns (uint256) {
        return totalRewardsDistributed;
    }
    
    // ========================================================================
    // ROUND MANAGEMENT
    // ========================================================================
    
    /**
     * @dev Start a new FL round
     */
    function startNewRound() public onlyOwner {
        currentRound += 1;
    }
    
    /**
     * @dev Complete current round with global model hash
     */
    function completeRound(string memory globalModelHash) public onlyOwner {
        require(
            roundUpdates[currentRound].length > 0,
            "No updates in this round"
        );
        
        // Record final global model hash
        ModelUpdate memory globalUpdate = ModelUpdate({
            round: currentRound,
            hospital: owner,
            modelHash: globalModelHash,
            timestamp: block.timestamp,
            verified: true,
            accuracy: 0,
            updateDetails: "Global model"
        });
        
        roundUpdates[currentRound].push(globalUpdate);
        
        emit RoundCompleted(
            currentRound,
            registeredHospitals.length,
            globalModelHash
        );
    }
    
    /**
     * @dev Get current round number
     */
    function getCurrentRound() public view returns (uint256) {
        return currentRound;
    }
    
    // ========================================================================
    // AUDIT & TRANSPARENCY
    // ========================================================================
    
    /**
     * @dev Get audit trail for a hospital
     */
    function getHospitalAuditTrail(address hospital)
        public
        view
        returns (ModelUpdate[] memory)
    {
        // Count updates from hospital
        uint256 count = 0;
        for (uint256 round = 1; round <= currentRound; round++) {
            for (uint256 i = 0; i < roundUpdates[round].length; i++) {
                if (roundUpdates[round][i].hospital == hospital) {
                    count++;
                }
            }
        }
        
        // Collect updates
        ModelUpdate[] memory auditTrail = new ModelUpdate[](count);
        uint256 index = 0;
        for (uint256 round = 1; round <= currentRound; round++) {
            for (uint256 i = 0; i < roundUpdates[round].length; i++) {
                if (roundUpdates[round][i].hospital == hospital) {
                    auditTrail[index] = roundUpdates[round][i];
                    index++;
                }
            }
        }
        
        return auditTrail;
    }
    
    /**
     * @dev Get network statistics
     */
    function getNetworkStats()
        public
        view
        returns (
            uint256 totalHospitals,
            uint256 activeHospitals,
            uint256 currentRoundNumber,
            uint256 totalRewards
        )
    {
        uint256 active = 0;
        for (uint256 i = 0; i < registeredHospitals.length; i++) {
            if (hospitals[registeredHospitals[i]].isActive) {
                active++;
            }
        }
        
        return (
            registeredHospitals.length,
            active,
            currentRound,
            totalRewardsDistributed
        );
    }
    
    // ========================================================================
    // EMERGENCY FUNCTIONS
    // ========================================================================
    
    /**
     * @dev Suspend a hospital due to suspicious activity
     */
    function suspendHospital(address hospital, string memory reason)
        public
        onlyOwner
    {
        require(hospitals[hospital].isActive, "Hospital already inactive");
        hospitals[hospital].isActive = false;
        
        emit ReputationChanged(
            hospital,
            hospitals[hospital].reputation,
            0,
            string(abi.encodePacked("Suspended: ", reason))
        );
    }
    
    /**
     * @dev Reactivate a suspended hospital
     */
    function reactivateHospital(address hospital) public onlyOwner {
        require(!hospitals[hospital].isActive, "Hospital already active");
        hospitals[hospital].isActive = true;
        hospitals[hospital].reputation = 50;  // Reset reputation
    }
}

/**
 * @title FLDataGovernance
 * @dev Manages data governance and consent records
 */
contract FLDataGovernance {
    
    struct PatientConsent {
        address patient;
        bool consentGiven;
        uint256 consentTime;
        uint256 expiryTime;
        string consentDetails;
    }
    
    mapping(address => PatientConsent) public patientConsents;
    address public owner;
    
    event ConsentRecorded(
        address indexed patient,
        uint256 consentTime,
        uint256 expiryTime
    );
    
    event ConsentRevoked(address indexed patient, uint256 revokeTime);
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Record patient consent
     */
    function recordConsent(
        address patient,
        uint256 durationDays,
        string memory details
    ) public {
        require(msg.sender == owner, "Only owner can record consent");
        
        uint256 expiryTime = block.timestamp + (durationDays * 1 days);
        
        patientConsents[patient] = PatientConsent({
            patient: patient,
            consentGiven: true,
            consentTime: block.timestamp,
            expiryTime: expiryTime,
            consentDetails: details
        });
        
        emit ConsentRecorded(patient, block.timestamp, expiryTime);
    }
    
    /**
     * @dev Check if consent is valid
     */
    function isConsentValid(address patient) public view returns (bool) {
        PatientConsent memory consent = patientConsents[patient];
        return consent.consentGiven && (block.timestamp < consent.expiryTime);
    }
    
    /**
     * @dev Revoke consent
     */
    function revokeConsent(address patient) public {
        require(msg.sender == owner, "Only owner can revoke consent");
        patientConsents[patient].consentGiven = false;
        emit ConsentRevoked(patient, block.timestamp);
    }
}
