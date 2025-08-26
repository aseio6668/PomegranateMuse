"""
Security Integration Module for MyndraComposer
Integrates security analysis with code generation and project management
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .vulnerability_scanner import (
    SecurityScanner, SecurityScanResult, SecurityVulnerability,
    SeverityLevel, VulnerabilityCategory
)


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    fail_on_critical: bool = True
    fail_on_high: bool = False
    max_risk_score: int = 50
    excluded_categories: List[str] = None
    excluded_files: List[str] = None
    custom_rules: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.excluded_categories is None:
            self.excluded_categories = []
        if self.excluded_files is None:
            self.excluded_files = []
        if self.custom_rules is None:
            self.custom_rules = []


@dataclass
class SecurityGate:
    """Security gate for CI/CD integration"""
    name: str
    policy: SecurityPolicy
    enabled: bool = True
    auto_fix: bool = False
    notify_on_failure: bool = True


class SecurityIntegration:
    """Main security integration class"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.scanner = SecurityScanner(
            results_dir=str(self.project_root / ".pomuse" / "security")
        )
        self.config_file = self.project_root / ".pomuse" / "security_config.json"
        self.policy = self._load_security_policy()
        self.security_gates = self._load_security_gates()
    
    def _load_security_policy(self) -> SecurityPolicy:
        """Load security policy from configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    policy_data = config_data.get("policy", {})
                    return SecurityPolicy(**policy_data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load security policy: {e}")
        
        # Return default policy
        return SecurityPolicy()
    
    def _load_security_gates(self) -> Dict[str, SecurityGate]:
        """Load security gates configuration"""
        gates = {}
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    gates_data = config_data.get("gates", {})
                    
                    for gate_name, gate_config in gates_data.items():
                        policy_data = gate_config.pop("policy", {})
                        policy = SecurityPolicy(**policy_data)
                        gate_config["policy"] = policy
                        gates[gate_name] = SecurityGate(name=gate_name, **gate_config)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load security gates: {e}")
        
        # Create default gates if none exist
        if not gates:
            gates = self._create_default_gates()
        
        return gates
    
    def _create_default_gates(self) -> Dict[str, SecurityGate]:
        """Create default security gates"""
        return {
            "pre_commit": SecurityGate(
                name="pre_commit",
                policy=SecurityPolicy(
                    fail_on_critical=True,
                    fail_on_high=False,
                    max_risk_score=30
                ),
                auto_fix=False
            ),
            "ci_build": SecurityGate(
                name="ci_build",
                policy=SecurityPolicy(
                    fail_on_critical=True,
                    fail_on_high=True,
                    max_risk_score=20
                ),
                auto_fix=False
            ),
            "pre_production": SecurityGate(
                name="pre_production",
                policy=SecurityPolicy(
                    fail_on_critical=True,
                    fail_on_high=True,
                    max_risk_score=10
                ),
                auto_fix=False,
                notify_on_failure=True
            )
        }
    
    def save_security_config(self):
        """Save security configuration to file"""
        config_data = {
            "policy": asdict(self.policy),
            "gates": {}
        }
        
        for gate_name, gate in self.security_gates.items():
            gate_data = asdict(gate)
            gate_data.pop("name")  # Remove redundant name field
            config_data["gates"][gate_name] = gate_data
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    async def scan_project_security(
        self, 
        include_external: bool = True,
        policy_check: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive security scan with policy evaluation"""
        
        print("🔒 Running security analysis...")
        
        # Run security scan
        scan_result = await self.scanner.scan_project(
            str(self.project_root), 
            include_external=include_external
        )
        
        if not scan_result.success:
            return {
                "success": False,
                "error": scan_result.error_details,
                "scan_result": scan_result
            }
        
        # Evaluate against security policy
        policy_result = None
        if policy_check:
            policy_result = self.evaluate_security_policy(
                scan_result.vulnerabilities, 
                self.policy
            )
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(scan_result)
        
        return {
            "success": True,
            "scan_result": scan_result,
            "policy_result": policy_result,
            "recommendations": recommendations,
            "summary": {
                "total_vulnerabilities": len(scan_result.vulnerabilities),
                "risk_score": scan_result.summary.get("risk_score", 0),
                "policy_compliant": policy_result["compliant"] if policy_result else None,
                "files_scanned": scan_result.total_files_scanned
            }
        }
    
    def evaluate_security_policy(
        self, 
        vulnerabilities: List[SecurityVulnerability], 
        policy: SecurityPolicy
    ) -> Dict[str, Any]:
        """Evaluate vulnerabilities against security policy"""
        
        # Filter vulnerabilities based on policy exclusions
        filtered_vulns = self._filter_vulnerabilities(vulnerabilities, policy)
        
        # Check policy violations
        violations = []
        
        # Check for critical vulnerabilities
        critical_vulns = [v for v in filtered_vulns if v.severity == SeverityLevel.CRITICAL]
        if policy.fail_on_critical and critical_vulns:
            violations.append({
                "type": "critical_vulnerabilities",
                "count": len(critical_vulns),
                "message": f"Found {len(critical_vulns)} critical vulnerabilities"
            })
        
        # Check for high vulnerabilities
        high_vulns = [v for v in filtered_vulns if v.severity == SeverityLevel.HIGH]
        if policy.fail_on_high and high_vulns:
            violations.append({
                "type": "high_vulnerabilities",
                "count": len(high_vulns),
                "message": f"Found {len(high_vulns)} high severity vulnerabilities"
            })
        
        # Check risk score
        total_risk_score = self._calculate_risk_score(filtered_vulns)
        if total_risk_score > policy.max_risk_score:
            violations.append({
                "type": "risk_score_exceeded",
                "score": total_risk_score,
                "limit": policy.max_risk_score,
                "message": f"Risk score {total_risk_score} exceeds limit of {policy.max_risk_score}"
            })
        
        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "filtered_vulnerabilities": len(filtered_vulns),
            "total_vulnerabilities": len(vulnerabilities),
            "risk_score": total_risk_score,
            "policy": asdict(policy)
        }
    
    def _filter_vulnerabilities(
        self, 
        vulnerabilities: List[SecurityVulnerability], 
        policy: SecurityPolicy
    ) -> List[SecurityVulnerability]:
        """Filter vulnerabilities based on policy exclusions"""
        filtered = []
        
        for vuln in vulnerabilities:
            # Check category exclusions
            if vuln.category.value in policy.excluded_categories:
                continue
            
            # Check file exclusions
            excluded = False
            for excluded_pattern in policy.excluded_files:
                if excluded_pattern in vuln.file_path:
                    excluded = True
                    break
            
            if not excluded:
                filtered.append(vuln)
        
        return filtered
    
    def _calculate_risk_score(self, vulnerabilities: List[SecurityVulnerability]) -> int:
        """Calculate total risk score for vulnerabilities"""
        severity_weights = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 7,
            SeverityLevel.MEDIUM: 4,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1
        }
        
        return sum(
            severity_weights.get(vuln.severity, 1) 
            for vuln in vulnerabilities
        )
    
    async def run_security_gate(self, gate_name: str) -> Dict[str, Any]:
        """Run a specific security gate"""
        if gate_name not in self.security_gates:
            return {
                "success": False,
                "error": f"Security gate '{gate_name}' not found"
            }
        
        gate = self.security_gates[gate_name]
        
        if not gate.enabled:
            return {
                "success": True,
                "skipped": True,
                "message": f"Security gate '{gate_name}' is disabled"
            }
        
        print(f"🚪 Running security gate: {gate_name}")
        
        # Run security scan
        scan_result = await self.scanner.scan_project(str(self.project_root))
        
        if not scan_result.success:
            return {
                "success": False,
                "error": scan_result.error_details,
                "gate": gate_name
            }
        
        # Evaluate against gate policy
        policy_result = self.evaluate_security_policy(
            scan_result.vulnerabilities, 
            gate.policy
        )
        
        gate_passed = policy_result["compliant"]
        
        result = {
            "success": gate_passed,
            "gate": gate_name,
            "scan_result": scan_result,
            "policy_result": policy_result,
            "violations": policy_result["violations"],
            "auto_fix_attempted": False
        }
        
        # Auto-fix if enabled and gate failed
        if not gate_passed and gate.auto_fix:
            print(f"🔧 Attempting auto-fix for gate: {gate_name}")
            auto_fix_result = await self._attempt_auto_fix(scan_result.vulnerabilities)
            result["auto_fix_attempted"] = True
            result["auto_fix_result"] = auto_fix_result
        
        # Notify on failure if enabled
        if not gate_passed and gate.notify_on_failure:
            await self._notify_security_failure(gate_name, policy_result)
        
        return result
    
    async def _attempt_auto_fix(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, Any]:
        """Attempt to automatically fix security vulnerabilities"""
        fixes_applied = []
        fixes_failed = []
        
        for vuln in vulnerabilities:
            try:
                fix_result = await self._fix_vulnerability(vuln)
                if fix_result["success"]:
                    fixes_applied.append({
                        "vulnerability_id": vuln.id,
                        "fix_description": fix_result["description"]
                    })
                else:
                    fixes_failed.append({
                        "vulnerability_id": vuln.id,
                        "error": fix_result["error"]
                    })
            except Exception as e:
                fixes_failed.append({
                    "vulnerability_id": vuln.id,
                    "error": str(e)
                })
        
        return {
            "fixes_applied": len(fixes_applied),
            "fixes_failed": len(fixes_failed),
            "applied_fixes": fixes_applied,
            "failed_fixes": fixes_failed
        }
    
    async def _fix_vulnerability(self, vulnerability: SecurityVulnerability) -> Dict[str, Any]:
        """Attempt to fix a specific vulnerability"""
        # This is a placeholder for auto-fix functionality
        # In a real implementation, this would contain specific fix logic
        # for different vulnerability types
        
        if vulnerability.category == VulnerabilityCategory.SENSITIVE_DATA:
            return await self._fix_hardcoded_secret(vulnerability)
        elif vulnerability.category == VulnerabilityCategory.CRYPTOGRAPHIC_FAILURES:
            return await self._fix_weak_crypto(vulnerability)
        else:
            return {
                "success": False,
                "error": f"No auto-fix available for {vulnerability.category.value}"
            }
    
    async def _fix_hardcoded_secret(self, vulnerability: SecurityVulnerability) -> Dict[str, Any]:
        """Fix hardcoded secret vulnerability"""
        # Placeholder for secret remediation
        return {
            "success": False,
            "error": "Auto-fix for hardcoded secrets requires manual intervention"
        }
    
    async def _fix_weak_crypto(self, vulnerability: SecurityVulnerability) -> Dict[str, Any]:
        """Fix weak cryptography vulnerability"""
        # Placeholder for crypto fix
        return {
            "success": False,
            "error": "Auto-fix for cryptographic issues requires manual review"
        }
    
    async def _notify_security_failure(self, gate_name: str, policy_result: Dict[str, Any]):
        """Notify about security gate failure"""
        # Placeholder for notification system
        print(f"🚨 Security gate '{gate_name}' failed!")
        for violation in policy_result["violations"]:
            print(f"   - {violation['message']}")
    
    def _generate_security_recommendations(self, scan_result: SecurityScanResult) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []
        
        if not scan_result.vulnerabilities:
            recommendations.append("✅ No security vulnerabilities detected!")
            return recommendations
        
        # Categorize recommendations by severity
        critical_count = sum(1 for v in scan_result.vulnerabilities if v.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for v in scan_result.vulnerabilities if v.severity == SeverityLevel.HIGH)
        
        if critical_count > 0:
            recommendations.append(
                f"🚨 Immediate action required: {critical_count} critical vulnerabilities"
            )
        
        if high_count > 0:
            recommendations.append(
                f"⚠️  High priority: {high_count} high severity vulnerabilities"
            )
        
        # Category-specific recommendations
        category_counts = {}
        for vuln in scan_result.vulnerabilities:
            category = vuln.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Top vulnerability categories
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for category, count in top_categories:
            if count > 2:
                category_name = category.replace('_', ' ').title()
                recommendations.append(
                    f"🔍 Focus area: {count} {category_name} vulnerabilities detected"
                )
        
        # Tool-specific recommendations
        if "semgrep" not in scan_result.tools_used:
            recommendations.append("🛠️  Consider installing Semgrep for enhanced static analysis")
        
        if "bandit" not in scan_result.tools_used and self._has_python_files():
            recommendations.append("🐍 Consider installing Bandit for Python security analysis")
        
        return recommendations
    
    def _has_python_files(self) -> bool:
        """Check if project contains Python files"""
        return len(list(self.project_root.rglob("*.py"))) > 0
    
    def create_security_gate(self, gate_name: str, policy: SecurityPolicy) -> bool:
        """Create a new security gate"""
        try:
            gate = SecurityGate(name=gate_name, policy=policy)
            self.security_gates[gate_name] = gate
            self.save_security_config()
            print(f"✅ Created security gate: {gate_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to create security gate: {e}")
            return False
    
    def list_security_gates(self) -> Dict[str, Dict[str, Any]]:
        """List all security gates"""
        gates_info = {}
        
        for gate_name, gate in self.security_gates.items():
            gates_info[gate_name] = {
                "enabled": gate.enabled,
                "auto_fix": gate.auto_fix,
                "notify_on_failure": gate.notify_on_failure,
                "policy": {
                    "fail_on_critical": gate.policy.fail_on_critical,
                    "fail_on_high": gate.policy.fail_on_high,
                    "max_risk_score": gate.policy.max_risk_score
                }
            }
        
        return gates_info
    
    def get_security_dashboard(self, days: int = 30) -> Dict[str, Any]:
        """Get security dashboard data"""
        # Get recent scan results
        recent_scans = []
        scan_files = list((self.project_root / ".pomuse" / "security").glob("*.json"))
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for scan_file in scan_files:
            try:
                with open(scan_file, 'r') as f:
                    scan_data = json.load(f)
                
                scan_date = datetime.fromisoformat(scan_data["timestamp"])
                if scan_date > cutoff_date:
                    recent_scans.append(scan_data)
            except Exception:
                continue
        
        # Calculate dashboard metrics
        total_scans = len(recent_scans)
        total_vulns = sum(len(scan.get("vulnerabilities", [])) for scan in recent_scans)
        
        # Risk trend
        risk_trend = []
        for scan in sorted(recent_scans, key=lambda x: x["timestamp"]):
            risk_trend.append({
                "date": scan["timestamp"][:10],
                "risk_score": scan.get("summary", {}).get("risk_score", 0),
                "vulnerability_count": len(scan.get("vulnerabilities", []))
            })
        
        # Current security posture
        latest_scan = max(recent_scans, key=lambda x: x["timestamp"]) if recent_scans else None
        current_posture = "unknown"
        
        if latest_scan:
            latest_risk = latest_scan.get("summary", {}).get("risk_score", 0)
            if latest_risk == 0:
                current_posture = "excellent"
            elif latest_risk < 20:
                current_posture = "good"
            elif latest_risk < 50:
                current_posture = "moderate"
            else:
                current_posture = "poor"
        
        return {
            "period": f"Last {days} days",
            "total_scans": total_scans,
            "total_vulnerabilities": total_vulns,
            "current_posture": current_posture,
            "risk_trend": risk_trend,
            "security_gates": self.list_security_gates(),
            "latest_scan": latest_scan
        }


# CLI interface functions
async def run_security_scan_interactive(project_root: str):
    """Interactive security scan runner"""
    integration = SecurityIntegration(project_root)
    
    print("🔒 MyndraComposer Security Analysis")
    print("=" * 50)
    
    # Scan options
    print("\nScan options:")
    print("1. Quick scan (static analysis only)")
    print("2. Comprehensive scan (static + external tools)")
    print("3. Security gate check")
    print("4. Security dashboard")
    print("5. Configure security policy")
    
    try:
        choice = int(input("\nSelect option (1-5): "))
        
        if choice == 1:
            print("\n🔍 Running quick security scan...")
            result = await integration.scan_project_security(include_external=False)
            
            if result["success"]:
                summary = result["summary"]
                print(f"\n✅ Quick scan completed!")
                print(f"   Vulnerabilities: {summary['total_vulnerabilities']}")
                print(f"   Risk Score: {summary['risk_score']}")
                print(f"   Files Scanned: {summary['files_scanned']}")
            else:
                print(f"❌ Scan failed: {result['error']}")
        
        elif choice == 2:
            print("\n🔍 Running comprehensive security scan...")
            result = await integration.scan_project_security(include_external=True)
            
            if result["success"]:
                summary = result["summary"]
                policy_result = result["policy_result"]
                
                print(f"\n✅ Comprehensive scan completed!")
                print(f"   Vulnerabilities: {summary['total_vulnerabilities']}")
                print(f"   Risk Score: {summary['risk_score']}")
                print(f"   Policy Compliant: {summary['policy_compliant']}")
                
                if policy_result and not policy_result["compliant"]:
                    print("\n⚠️  Policy violations:")
                    for violation in policy_result["violations"]:
                        print(f"     - {violation['message']}")
                
                print("\n💡 Recommendations:")
                for rec in result["recommendations"]:
                    print(f"   {rec}")
            else:
                print(f"❌ Scan failed: {result['error']}")
        
        elif choice == 3:
            gates = integration.list_security_gates()
            print("\n🚪 Available security gates:")
            gate_names = list(gates.keys())
            
            for i, gate_name in enumerate(gate_names, 1):
                gate_info = gates[gate_name]
                status = "enabled" if gate_info["enabled"] else "disabled"
                print(f"   {i}. {gate_name} ({status})")
            
            gate_choice = int(input(f"\nSelect gate to run (1-{len(gate_names)}): "))
            if 1 <= gate_choice <= len(gate_names):
                selected_gate = gate_names[gate_choice - 1]
                
                result = await integration.run_security_gate(selected_gate)
                
                if result.get("skipped"):
                    print(f"⏭️  {result['message']}")
                elif result["success"]:
                    print(f"✅ Security gate '{selected_gate}' passed!")
                else:
                    print(f"❌ Security gate '{selected_gate}' failed!")
                    for violation in result.get("violations", []):
                        print(f"   - {violation['message']}")
        
        elif choice == 4:
            days = int(input("Dashboard period in days [30]: ") or "30")
            dashboard = integration.get_security_dashboard(days)
            
            print(f"\n📊 Security Dashboard ({dashboard['period']})")
            print(f"   Total Scans: {dashboard['total_scans']}")
            print(f"   Total Vulnerabilities: {dashboard['total_vulnerabilities']}")
            print(f"   Current Posture: {dashboard['current_posture']}")
            
            print("\n🚪 Security Gates:")
            for gate_name, gate_info in dashboard["security_gates"].items():
                status = "✅" if gate_info["enabled"] else "❌"
                print(f"   {status} {gate_name}")
        
        elif choice == 5:
            print("\n⚙️  Security policy configuration")
            print("Current policy settings:")
            print(f"   Fail on Critical: {integration.policy.fail_on_critical}")
            print(f"   Fail on High: {integration.policy.fail_on_high}")
            print(f"   Max Risk Score: {integration.policy.max_risk_score}")
            
            # Allow user to modify policy
            fail_critical = input("Fail on critical vulnerabilities? [Y/n]: ").strip().lower()
            if fail_critical != 'n':
                integration.policy.fail_on_critical = True
            
            fail_high = input("Fail on high severity vulnerabilities? [y/N]: ").strip().lower()
            integration.policy.fail_on_high = (fail_high == 'y')
            
            max_risk = input(f"Maximum risk score [{integration.policy.max_risk_score}]: ").strip()
            if max_risk:
                integration.policy.max_risk_score = int(max_risk)
            
            integration.save_security_config()
            print("✅ Security policy updated!")
        
    except (ValueError, KeyboardInterrupt):
        print("\nSecurity analysis cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(run_security_scan_interactive(project_path))